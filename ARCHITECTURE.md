# RAG 汽车知识问答系统 · 架构设计文档

> 基于吉利汽车用户手册的检索增强生成（RAG）问答系统  
> 核心技术栈：多路召回 · Cross-Encoder 重排 · 本地/云端 LLM · LangChain · FAISS

---

## 目录

1. [系统概述](#1-系统概述)
2. [整体架构](#2-整体架构)
3. [核心模块详解](#3-核心模块详解)
4. [完整数据流](#4-完整数据流)
5. [设计亮点与权衡](#5-设计亮点与权衡)
6. [模块依赖关系](#6-模块依赖关系)
7. [配置与部署](#7-配置与部署)
8. [已知问题与优化建议](#8-已知问题与优化建议)

---

## 1. 系统概述

### 1.1 项目定位

本系统是一个基于 **RAG（Retrieval-Augmented Generation）** 架构的垂直领域知识问答系统，专注于回答汽车（吉利）使用、维修、保养等相关问题。

系统以汽车用户手册 PDF 为唯一知识来源，通过：
1. **多粒度 PDF 解析** → 构建高质量知识库
2. **多路异构检索** → 同时覆盖语义相关性和关键词相关性
3. **Cross-Encoder 精排** → 从候选集中挑选最相关文档
4. **大模型生成** → 基于检索到的知识生成自然语言答案

### 1.2 支持的 LLM 基座

| 模型 | 类型 | 参数量 | 部署方式 |
|------|------|--------|----------|
| Qwen2-7B-Instruct | 本地 HF | 7B | `hf_model.ChatLLM` |
| Baichuan2-7B-Chat | 本地 HF | 7B | `hf_model.ChatLLM` |
| ChatGLM3-6B | 本地 HF | 6B | `hf_model.ChatLLM` |
| Qwen3-9B / Qwen3.5-9B | 云端 API | 9B | `huggingface_proxy.ChatGPTProxy` |

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 汽车知识问答系统                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   入口 / 控制层                       │   │
│  │   run.py (批量评测)    example_test.py (交互问答)    │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              核心调度层 (generate_answer.py)          │   │
│  │   Prompt增强 → 四路召回 → 重排 → Prompt构建 → 推理  │   │
│  └────┬──────────┬──────────────────┬────────────┬──────┘   │
│       │          │                  │            │          │
│  ┌────▼──┐  ┌────▼────┐       ┌────▼────┐  ┌────▼────┐   │
│  │pdf_   │  │retriever│       │rerank_  │  │hf_model │   │
│  │parse  │  │ (4路)   │       │model    │  │/hf_proxy│   │
│  └───────┘  └─────────┘       └─────────┘  └─────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              评测层 (test_score.py)                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 `config.py` — 全局配置

集中管理所有设备配置与模型路径，被几乎所有模块引用。

```python
# 设备自动检测
EMBEDDING_DEVICE = "cuda" if cuda else "mps" if mps else "cpu"
LLM_DEVICE       = "cuda" if cuda else "mps" if mps else "cpu"

# 模型路径
Qwen2_path      = './models/Qwen2-7B-Instruct'
Baichuan_path   = './models/Baichuan2-7B-Chat'
ChatGLM_path    = './models/chatglm3-6b'

M3E_embeddings_model_path = "./pre_train_model/m3e-large"
BGE_embeddings_model_path = "./pre_train_model/bge-m3"
BGE_reranker_model        = "./pre_train_model/bge-reranker-large"
BCE_reranker_model        = "./pre_train_model/bce-reranker-base_v1"
SimModel_path             = './pre_train_model/text2vec-base-chinese'
```

---

### 3.2 `pdf_parse.py` — 多策略 PDF 解析器

**设计目标**：最大化文本块的覆盖率、完整性和多粒度性，避免单一解析策略导致的信息丢失。

**三种解析策略对比**：

| 策略 | 方法 | 工具 | 优势 | 适合场景 |
|------|------|------|------|----------|
| `ParseBlock` | 分块解析法 | pdfplumber | 保证小标题+内容在同一块 | 结构化内容，带标题的段落 |
| `ParseAllPage` | 滑动窗口法 | PyPDF2 | 保证跨页语义连续性 | 跨页延续的长段落 |
| `ParseOnePageWithRule` | 规则切分法 | PyPDF2 | 均匀切分，简单可靠 | 简单格式的内容 |

**解析流程**：
```
PDF 文件
  → ParseBlock(1024) + ParseBlock(512)       ← 双粒度分块
  → ParseAllPage(256) + ParseAllPage(512)    ← 双粒度滑窗
  → ParseOnePageWithRule(256) + (512)        ← 双粒度规则
  → 去重合并
  → 最终文本块列表（数千条）
```

**`Datafilter` 清洗规则**：
- 过滤长度 < 6 的短文本
- 对超长文本按 `■` → `•` → `\t` → `。` 多级分割
- 去除 `\n`、`,`、`\t` 等噪声字符
- 全局去重

---

### 3.3 四路检索器（`retriever/`）

#### 稠密语义召回（向量检索）

两种向量模型互补，覆盖不同语义空间：

**M3E Retriever** (`m3e_retriever.py`)
```python
# moka-ai/m3e-large，专注中文语义
HuggingFaceEmbeddings(batch_size=64)
FAISS.from_documents(docs, embeddings)
→ similarity_search_with_score(query, k=15)  # 返回 (Document, L2_score)
```

**BGE Retriever** (`bge_retriever.py`)
```python
# BAAI/bge-m3，支持多语言，L2归一化
HuggingFaceBgeEmbeddings(normalize_embeddings=True)
FAISS.from_documents(docs, embeddings)
→ similarity_search_with_score(query, k=15)  # 返回 (Document, L2_score)
```

#### 稀疏关键词召回（词频检索）

**BM25 Retriever** (`bm25_retriever.py`)
```python
# jieba 分词 → BM25 词频统计
jieba.cut_for_search(text) → tokens
BM25Retriever.from_documents(docs)
→ get_relevant_documents(query)  # 关键词精确匹配
```
> ⚠️ **已知问题**：`GetBM25TopK` 内有 `break`，实际只返回1条文档

**TF-IDF Retriever** (`tfidf_retriever.py`)
```python
# jieba 分词 → TF-IDF 权重
TFIDFRetriever.from_documents(docs)
→ get_relevant_documents(query)  # 重要词汇加权匹配
```

---

### 3.4 `rerank_model.py` — Cross-Encoder 精排

**两阶段检索架构**（Bi-Encoder → Cross-Encoder）：

```
第一阶段（召回）：Bi-Encoder
  query → embedding → ANN 搜索 → 大量候选（速度快，精度一般）

第二阶段（精排）：Cross-Encoder
  (query, doc1) → 交叉编码 → 相关性分数
  (query, doc2) → 交叉编码 → 相关性分数
  ...
  → 按分数降序排列 → 取 Top-K（精度高，速度慢但候选少）
```

**核心代码逻辑**：
```python
def predict(self, query, docs):
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = self.tokenizer(pairs, padding=True, truncation=True,
                            max_length=512, return_tensors='pt').to("cuda")
    with torch.no_grad():
        scores = self.model(**inputs).logits
    # 按分数降序返回文档
    return [doc for score, doc in sorted(zip(scores, docs), reverse=True)]
```

---

### 3.5 `generate_answer.py` — RAG 核心流程

**`question_test` 主流程**：

```
初始化阶段：
  M3eRetriever + BgeRetriever + Bm25Retriever + TfidfRetriever
  ChatLLM (or ChatGPTProxy) + reRankLLM

对每条测试问题：

  Step 1 [可选] Query 增强
    prompt = "请简洁回答...{query}"（限16字）
    answer = llm.infer([prompt])
    retriever_query = query + answer

  Step 2 四路召回（各取15条）
    m3e_context  = m3e_retriever.GetTopK(retriever_query, 15)
    bge_context  = bge_retriever.GetTopK(retriever_query, 15)
    bm25_context = bm25.GetBM25TopK(retriever_query, 15)
    tfidf_context = tfidf.GetBM25TopK(retriever_query, 15)

  Step 3 构建5路 Prompt
    m3e_inputs        = get_emb_docs(m3e_context, query, top_k=6)
    bge_inputs        = get_emb_docs(bge_context, query, top_k=6)
    bm25_inputs       = get_distribute_docs(bm25_context, query, top_k=6)
    tfidf_inputs      = get_distribute_docs(tfidf_context, query, top_k=6)
    mutil_rerank_inputs = get_emb_distribute_rerank(
                            rerank, m3e+bge+bm25+tfidf, query, top_k=6)

  Step 4 Batch 推理
    batch_input  = [mutil_rerank, m3e, bge, bm25, tfidf]
    batch_output = llm.infer(batch_input)

  Step 5 兜底过滤
    if m3e_min_score > 500: answer_6 = "无答案"
    if bge_min_score > 500: answer_7 = "无答案"

  保存 result.json
```

**Prompt 模板**：
```
基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说"无答案"，不允许在答案中添加编造成分，答案请使用中文。
已知内容为吉利控股集团汽车销售有限公司的吉利用户手册：
{retriever_text}
问题: {question}
回答:
```

---

### 3.6 LLM 推理层

**本地推理** (`hf_model.py · ChatLLM`)
```python
# 三种模型对应不同的 Chat Template
if model_name == "qwen2":
    text = f"<|im_start|>system\n...<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
if model_name == "baichuan2":
    text = f"<reserved_106>{q}<reserved_107>"
if model_name == "chatglm3":
    text = f"<|system|>\n...<|user|>\n{q}\n<|assistant|>\n"

# 推理参数
max_new_tokens=2000, temperature=0.0, do_sample=False, repetition_penalty=1.05
```

**云端代理** (`huggingface_proxy.py · ChatGPTProxy`)
```python
# HuggingFace Inference Router（OpenAI 兼容接口）
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)

# 线程池并行推理（提升 batch 吞吐量）
with ThreadPoolExecutor() as executor:
    results = list(executor.map(self.get_response, prompts))
```

---

### 3.7 `test_score.py` — 评测评分

- 加载 `text2vec-base-chinese` 模型
- 对5种召回策略的答案分别与标准答案计算**语义相似度**
- 支持横向对比不同召回策略的实际效果
- 输出格式：`metrics.json`

---

## 4. 完整数据流

```
【阶段 0】环境准备
download_model.py
  → 下载 m3e-large / bge-m3 / bce-reranker / text2vec → pre_train_model/
  → 手动下载 LLM 权重 → models/

【阶段 1】知识库构建（首次运行自动触发）
car_user_manual.pdf
  → pdf_parse.DataProcess
      ├── ParseBlock(1024 / 512)
      ├── ParseAllPage(256 / 512)
      └── ParseOnePageWithRule(256 / 512)
      ↓ 去重合并（数千文本块）
  ├── M3eRetriever → HuggingFaceEmbeddings → FAISS → faiss_m3e_index/
  ├── BgeRetriever → HuggingFaceBgeEmbeddings → FAISS → faiss_bge_index/
  ├── Bm25Retriever → jieba → BM25Retriever（内存）
  └── TfidfRetriever → jieba → TFIDFRetriever（内存）

【阶段 2】Query 增强（可选）
query
  → LLM.infer("请简洁地回答...{query}")
  → retriever_query = query + answer（约 ≤16 字补充）

【阶段 3】四路并行召回
retriever_query
  ├── M3E.GetTopK(15)   → [(Document, L2_score), ...]   稠密语义
  ├── BGE.GetTopK(15)   → [(Document, L2_score), ...]   稠密语义
  ├── BM25.GetTopK(15)  → [Document, ...]               稀疏关键词
  └── TFIDF.GetTopK(15) → [Document, ...]               稀疏关键词
  合计：最多 60 条候选文档

【阶段 4】Cross-Encoder 精排
60 候选文档
  → reRankLLM.predict(query, docs)
      → tokenize (query, doc) pairs
      → Cross-Encoder 打分
      → 降序排列
  → 取 Top-6 最相关文档

【阶段 5】Prompt 构建（5路）
  mutil_rerank_inputs = Prompt(重排Top6, query)
  m3e_inputs          = Prompt(M3E Top6, query)
  bge_inputs          = Prompt(BGE Top6, query)
  bm25_inputs         = Prompt(BM25 Top6, query)
  tfidf_inputs        = Prompt(TFIDF Top6, query)

【阶段 6】Batch 推理
  batch_input = [mutil_rerank, m3e, bge, bm25, tfidf]
  → ChatLLM.infer(batch_input) 或 ChatGPTProxy.infer(batch_input)
  → batch_output = [answer_1, answer_2, answer_3, answer_4, answer_5]

【阶段 7】兜底过滤 + 保存
  if m3e_min_score > 500: answer_6 = "无答案"
  if bge_min_score > 500: answer_7 = "无答案"
  → 写入 result.json

【阶段 8】评测评分
result.json + gold_result.json
  → test_score.test_metrics
      → text2vec-base-chinese 语义相似度
  → 写入 metrics.json
```

---

## 5. 设计亮点与权衡

### 5.1 多路召回融合

| 召回方式 | 模型 | 优势 | 劣势 |
|----------|------|------|------|
| M3E 向量召回 | m3e-large | 中文语义理解强，泛化好 | 关键词命中弱 |
| BGE 向量召回 | bge-m3 | 多语言，精度高 | 计算成本高 |
| BM25 | rank-bm25 | 关键词精确匹配，速度快 | 语义理解差 |
| TF-IDF | sklearn | 重要词加权，简单有效 | 语义理解差 |

**设计权衡**：稠密召回（M3E/BGE）擅长语义泛化，稀疏召回（BM25/TF-IDF）擅长关键词精确匹配。四路融合后经 Cross-Encoder 精排，同时保障召回率和精准度。

### 5.2 两阶段检索（召回 + 精排）

```
Bi-Encoder（快速，精度一般）→ 大量候选（15条/路 × 4路 = 60条）
         ↓
Cross-Encoder（慢速，精度高）→ 精确排序 → Top-6 输入 LLM
```

**权衡**：若直接用 Cross-Encoder 全量检索，速度太慢；先用 Bi-Encoder 粗筛再精排，兼顾效率与精度。

### 5.3 多粒度 PDF 解析

6次解析（3策略 × 2粒度）+ 去重，确保：
- 标题+内容不被切断（ParseBlock）
- 跨页内容被完整捕获（ParseAllPage 滑窗）
- 不同检索粒度都能命中（256/512/1024 字符块）

### 5.4 Query 增强

先让 LLM 简短回答问题，将回答附加到查询词，扩充语义信息，改善以下场景：
- 模糊问题（"座椅太热" → LLM 补充"座椅加热功能"）
- 缩写/简写（LLM 补充全称）
- 跨模态问题（LLM 推断相关领域词汇）

### 5.5 5路对比评测

每道题保存5个答案，方便研究者：
- 直观对比不同召回策略效果
- 分析多路重排是否真正有提升
- 识别 BM25/TF-IDF 的适用场景

### 5.6 向量距离兜底

当最近邻距离 > 500 时，说明知识库中完全没有相关内容，直接输出"无答案"，避免 LLM 在无相关上下文时产生幻觉（Hallucination）。

---

## 6. 模块依赖关系

```
config.py
  ← hf_model.py
  ← rerank_model.py
  ← run.py
  ← test_score.py

pdf_parse.py
  ← retriever/m3e_retriever.py
  ← retriever/bge_retriever.py
  ← retriever/bm25_retriever.py
  ← retriever/tfidf_retriever.py

generate_answer.py
  ← run.py
  ← example_test.py
  → hf_model.py
  → huggingface_proxy.py
  → rerank_model.py
  → retriever/*.py

test_score.py
  ← run.py
  → config.py

benchmark/model_serve.py  （独立 FastAPI 服务）
download_model.py          （独立下载脚本）
```

---

## 7. 配置与部署

### 7.1 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/zhangzg1/rag_with_chat.git
cd rag_with_chat

# 2. 创建虚拟环境
conda create -n rag_with_chat python=3.9
conda activate rag_with_chat

# 3. 安装依赖
pip install -r requirements.txt
# 根据 CUDA 版本安装 torch：https://pytorch.org/get-started/locally/

# 4. 下载预训练模型
python download_model.py
# 手动下载 LLM 权重到 ./models/

# 5. 配置 API 密钥（使用云端 LLM 时）
echo "HF_TOKEN=hf_xxx" > .env

# 6. 运行 RAG 评测
python run.py --llm_name qwen2 --reranker_name bce

# 7. 交互式问答
python example_test.py
```

### 7.2 运行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm_name` | `qwen2` | LLM 选型（qwen2/baichuan2/chatglm3/Qwen API） |
| `--reranker_name` | `bce` | Reranker 选型（bce/bge） |
| `--prompt_enhance` | `True` | 是否启用 Query 增强 |
| `--single_top_k` | `6` | 单路召回取 Top-K 数量 |
| `--mutil_top_k` | `6` | 多路重排后取 Top-K 数量 |
| `--single_max_length` | `4000` | 单路召回最大文本长度 |
| `--mutil_max_length` | `4000` | 多路召回最大文本长度 |
| `--pdf_path` | `./data/car_user_manual.pdf` | 知识库 PDF 路径 |
| `--test_path` | `./data/test_question.json` | 测试集路径 |
| `--predict_path` | `./data/result.json` | 预测结果保存路径 |

### 7.3 目录结构

```
rag_with_chat/
├── benchmark/
│   ├── bench_data.json          # 压测数据
│   ├── benchmark.py             # 压力测试脚本
│   └── model_serve.py           # FastAPI 模型服务
├── data/
│   ├── car_user_manual.pdf      # 汽车用户手册（知识库）
│   ├── test_question.json       # 测试题集
│   └── gold_result.json         # 标准答案
├── models/                      # LLM 权重（手动下载）
│   ├── Qwen2-7B-Instruct/
│   ├── Baichuan2-7B-Chat/
│   └── chatglm3-6b/
├── pre_train_model/             # 预训练小模型（download_model.py 下载）
│   ├── m3e-large/
│   ├── bge-m3/
│   ├── bce-reranker-base_v1/
│   ├── bge-reranker-large/
│   └── text2vec-base-chinese/
├── retriever/
│   ├── bge_retriever.py
│   ├── bm25_retriever.py
│   ├── m3e_retriever.py
│   └── tfidf_retriever.py
├── vector_db/                   # FAISS 向量库（首次运行自动生成）
│   ├── faiss_m3e_index/
│   └── faiss_bge_index/
├── config.py
├── pdf_parse.py
├── rerank_model.py
├── generate_answer.py
├── hf_model.py
├── huggingface_proxy.py
├── test_score.py
├── example_test.py
├── run.py
├── download_model.py
├── requirements.txt
├── Dockerfile
└── .env                         # API 密钥（不提交 git）
```

### 7.4 Docker 部署

```bash
# 构建镜像（需先将模型下载到本地）
docker build -t rag_with_chat:latest .

# 启动评测
docker run --gpus all rag_with_chat:latest python run.py
```

---

## 8. 已知问题与优化建议

### 8.1 已知 Bug

| 文件 | 问题 | 影响 | 建议修复 |
|------|------|------|----------|
| `retriever/bm25_retriever.py` | `GetBM25TopK` 内有 `break`，只返回1条文档 | BM25 实际召回效果被严重削弱 | 删除 `break` 语句 |
| `example_test.py` | API 模型判断用 `"gpt" in model_name`，但实际使用 `"Qwen"` 判断 | 交互测试与批量测试行为不一致 | 统一判断逻辑 |

### 8.2 优化建议

#### 性能优化
- **向量库构建**：4个检索器都分别调用 PDF 解析，重复解析4次。建议先解析一次，将数据共享给所有检索器。
- **批量向量化**：M3E 已设置 `batch_size=64`，BGE 未设置，可补充提升速度。
- **vLLM 加速**：集成 vLLM 框架可获得约 1 倍推理速度提升（项目 README 提及，对应 `vllm_model.py`，已被删除，可补充）。

#### 效果优化
- **BM25 召回数量修复**：修复 `break` 问题，让 BM25 真正参与多路融合。
- **混合检索权重调整**：目前4路召回权重相同，可尝试根据评测结果动态调整各路权重。
- **Reranker 批量推理**：当前 Cross-Encoder 是对所有文档对逐一打分，可使用批处理加速。
- **上下文窗口利用**：当前 `max_length=4000`，可考虑按 LLM 实际支持的上下文窗口动态调整。

#### 工程优化
- **向量库持久化**：BM25 和 TF-IDF 每次启动都需重建，可序列化保存提升启动速度。
- **配置外部化**：将超参数（top_k、max_length、threshold）提取到配置文件，方便调参。
- **错误处理**：API 调用失败时仅返回错误字符串，建议增加重试机制。

---

*文档生成时间：2026-04-01 | 基于项目代码自动分析生成*
