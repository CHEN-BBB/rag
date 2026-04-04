# 吉利汽车智能问答系统（RAG with Chat）

> 基于检索增强生成（RAG）技术的汽车用户手册智能问答系统，融合四路召回 + 重排序 + 大语言模型，为用户提供精准的汽车知识问答服务。

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [模型准备](#模型准备)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [参数说明](#参数说明)
- [评估指标](#评估指标)
- [Docker 部署](#docker-部署)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 项目简介

本项目以**吉利控股集团汽车用户手册**为知识库，构建了一套完整的 RAG（Retrieval-Augmented Generation）知识问答系统。系统通过以下核心技术实现精准问答：

- **四路并行召回**：M3E（稠密语义）、BGE（稠密语义）、BM25（稀疏关键词）、TF-IDF（稀疏关键词）
- **两阶段检索**：Bi-Encoder 粗排召回 → Cross-Encoder 精排重排
- **双 LLM 路径**：支持本地 HuggingFace 模型推理 & 云端 API 调用
- **Prompt 增强**：利用 LLM 对用户问题进行查询扩展，提升召回质量
- **多维评估**：语义相似度 + 关键词匹配综合打分

---

## 系统架构

```
用户问题
    │
    ▼
[Prompt 增强] ──(LLM Query 扩展)──────────────────────────┐
    │                                                      │
    ▼                                                      │
[四路并行召回 Top-15/路]                                    │
  ├── M3E Retriever  (FAISS 稠密向量)                      │
  ├── BGE Retriever  (FAISS 稠密向量)                      │
  ├── BM25 Retriever (稀疏关键词)                          │
  └── TF-IDF Retriever (稀疏关键词)                        │
    │                                                      │
    ▼                                                      │
[Cross-Encoder 重排序 → Top-6]                             │
  ├── BCE Reranker                                         │
  └── BGE Reranker                                         │
    │                                                      │
    ▼                                                      │
[5路 Prompt 构建 → Batch LLM 推理]                         │
  ├── 多路重排结果 (answer_1) ◄──────────────────────────── ┘
  ├── M3E 单路结果 (answer_2)
  ├── BGE 单路结果 (answer_3)
  ├── BM25 单路结果 (answer_4)
  └── TF-IDF 单路结果 (answer_5)
    │
    ▼
[结果保存 + 评估打分]
```

---

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | 3.9+ |
| PyTorch | 2.1+（根据 CUDA 版本选择） |
| CUDA | 11.8 / 12.1（推荐，可使用 CPU 但速度较慢） |
| GPU 显存 | 本地模式建议 ≥ 16GB；API 模式无 GPU 要求 |
| 内存 | ≥ 16GB |

---

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/CHEN-BBB/rag.git
cd rag
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n rag_chat python=3.9
conda activate rag_chat

# 或使用 venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. 安装 PyTorch

> 请根据你的 CUDA 版本选择对应的安装命令，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)

```bash
# CUDA 11.8
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU Only（速度较慢）
pip install torch==2.1.2 torchvision torchaudio
```

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 5. 配置环境变量（API 模式必须）

如需使用 HuggingFace API 调用云端大模型，在项目根目录创建 `.env` 文件：

```bash
# .env
HF_TOKEN=your_huggingface_token_here
```

> 获取 HuggingFace Token：访问 https://huggingface.co/settings/tokens 创建 Access Token

---

## 模型准备

### 嵌入模型 & 重排模型（手动下载或在notebook中下载）

运行以下脚本，自动从 HuggingFace 下载所需的嵌入和重排模型到 `./pre_train_model/` 目录：

```bash
python download_model.py
```

下载的模型包括：

| 模型 | 用途 | 本地路径 |
|------|------|---------|
| `moka-ai/m3e-large` | M3E 稠密检索嵌入 | `./pre_train_model/m3e-large` |
| `BAAI/bge-m3` | BGE 稠密检索嵌入 | `./pre_train_model/bge-m3` |
| `InfiniFlow/bce-reranker-base_v1` | BCE 重排模型 | `./pre_train_model/bce-reranker-base_v1` |
| `shibing624/text2vec-base-chinese` | 评估相似度模型 | `./pre_train_model/text2vec-base-chinese` |

> 国内用户如遇下载困难，可使用 ModelScope 镜像或手动下载后放置到对应目录。

### 大语言模型（LLM）

根据使用模式选择：

#### 方式 A：本地部署 HuggingFace 模型

从 ModelScope 或 HuggingFace 下载模型到 `./models/` 目录：

```bash
# 以 Qwen2.5-7B-Instruct 为例（从 ModelScope 下载）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2.5-7B-Instruct', local_dir='./models/Qwen2.5-7B-Instruct')
"
```

支持的本地模型：

| 模型名 | 参数 `--llm_name` | 本地路径 |
|--------|------------------|---------|
| Qwen2.5-7B-Instruct | `qwen2` | `./models/Qwen2.5-7B-Instruct` |
| Baichuan2-7B-Chat | `baichuan2` | `./models/Baichuan2-7B-Chat` |
| ChatGLM3-6B | `chatglm3` | `./models/chatglm3-6b` |

#### 方式 B：HuggingFace Cloud API（无需 GPU）

无需下载模型，配置 `.env` 中的 `HF_TOKEN` 后，直接使用 API 调用：

```bash
# 使用 Qwen3-9B API
python run.py --llm_name "Qwen/Qwen3.5-9B"
```

---

## 快速开始

### 方式一：交互式问答（推荐入门）

适合直接体验系统效果，输入问题即可获得答案：

```bash
# all_text.txt 是 PDF 解析后的预处理文本文件，用于加速检索器的初始化。
python generate_all_text.py
# 使用本地 Qwen2 模型
python example_test.py

# 或修改 example_test.py 中的 model_name 参数后运行
```

启动后进入交互模式：

```
LLM model load ok
Retriever load ok
rerank model load ok
请输入问题（输入 'exit' 退出）：吉利汽车如何开启自动驾驶功能？
query:  吉利汽车如何开启自动驾驶功能？
answer:  根据用户手册，开启自动驾驶功能需要...
====================================================================================================
请输入问题（输入 'exit' 退出）：exit
退出程序。
```
### 方式二：使用notebook批量测试集评估

对测试集进行批量问答并自动评分：

```bash
#手动运行colab.ipynb
```
### 方式三：使用run.py批量测试集评估

对测试集进行批量问答并自动评分：

```bash
# 使用本地 Qwen2 + BCE 重排（默认配置）
python run.py

# 使用 BGE 重排
python run.py --llm_name qwen2 --reranker_name bge

# 使用 HuggingFace API（无需 GPU）
python run.py --llm_name "Qwen/Qwen3-9B"

# 禁用 Prompt 增强（提速但可能影响效果）
python run.py --prompt_enhance False
```

---

## 使用示例

### 示例 1：基本问答

```python
from huggingface_proxy import ChatGPTProxy
from rerank_model import reRankLLM
from retriever.m3e_retriever import M3eRetriever
from retriever.bge_retriever import BgeRetriever
from retriever.bm25_retriever import Bm25Retriever
from retriever.tfidf_retriever import TfidfRetriever
from generate_answer import get_emb_distribute_rerank

# 初始化模型（需要先准备好模型文件）
llm = ChatGPTProxy(model="Qwen/Qwen3.5-9B")  # 或使用本地模型 ChatLLM("qwen2")

m3e_retriever = M3eRetriever("./pre_train_model/m3e-large", pdf_path="./data/car_user_manual.pdf")
bge_retriever = BgeRetriever("./pre_train_model/bge-m3", pdf_path="./data/car_user_manual.pdf")
bm25 = Bm25Retriever(pdf_path="./data/car_user_manual.pdf")
tfidf = TfidfRetriever(pdf_path="./data/car_user_manual.pdf")
rerank = reRankLLM("bce")

# 提问
query = "吉利汽车的语音助手如何唤醒？"

# 四路召回
m3e_context = m3e_retriever.GetTopK(query, 15)
bge_context = bge_retriever.GetTopK(query, 15)
bm25_context = bm25.GetBM25TopK(query, 15)
tfidf_context = tfidf.GetBM25TopK(query, 15)

# 重排 + 生成答案
prompt = get_emb_distribute_rerank(rerank, m3e_context, bge_context, bm25_context, tfidf_context, query)
answer = llm.infer([prompt])
print(answer[0])
```

### 示例 2：仅使用 API 模式（无 GPU 环境）

```python
import os
os.environ["HF_TOKEN"] = "your_hf_token"  # 或在 .env 文件中配置

from huggingface_proxy import ChatGPTProxy

llm = ChatGPTProxy(model="Qwen/Qwen3.5-9B", temperature=0.1)
results = llm.infer([
    "吉利汽车发动机保养周期是多久？",
    "如何更换吉利汽车轮胎？"
])
for r in results:
    print(r)
```

### 示例 3：自定义 PDF 知识库

替换 `./data/car_user_manual.pdf` 为你自己的 PDF 文件，修改运行参数：

```bash
python run.py \
  --pdf_path ./data/my_manual.pdf \
  --llm_name qwen2 \
  --reranker_name bce \
  --mutil_top_k 8 \
  --single_max_length 5000
```

---

## 参数说明

运行 `python run.py --help` 查看所有可用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm_name` | `qwen2` | 使用的大语言模型，可选：`qwen2`、`baichuan2`、`chatglm3`、`Qwen/Qwen3-9B` |
| `--reranker_name` | `bce` | 重排模型，可选：`bce`、`bge` |
| `--prompt_enhance` | `True` | 是否启用 Prompt 增强（Query 扩展） |
| `--single_max_length` | `4000` | 单路召回最大文本长度（字符数） |
| `--single_top_k` | `6` | 单路召回返回的最大文档数 |
| `--mutil_max_length` | `4000` | 多路召回重排后最大文本长度 |
| `--mutil_top_k` | `6` | 多路召回重排后返回的最大文档数 |
| `--pdf_path` | `./data/car_user_manual.pdf` | 知识库 PDF 文件路径 |
| `--test_path` | `./data/test_question.json` | 测试问题集路径 |
| `--predict_path` | `./data/result.json` | 预测结果输出路径 |
| `--gold_path` | `./data/gold_result.json` | 标准答案路径 |
| `--metric_path` | `./data/metrics.json` | 评估指标输出路径 |
| `--m3e_embeddings_model` | `./pre_train_model/m3e-large` | M3E 嵌入模型路径 |
| `--bge_embeddings_model` | `./pre_train_model/bge-m3` | BGE 嵌入模型路径 |
| `--simModel_path` | `./pre_train_model/text2vec-base-chinese` | 评估相似度模型路径 |

---

## 评估指标

系统使用以下综合评分方法（`test_score.py`）：

```
综合得分 = 0.5 × 语义相似度得分 + 0.5 × 关键词匹配得分
```

- **语义相似度**：使用 `text2vec-base-chinese` 计算预测答案与标准答案的余弦相似度
- **关键词匹配**：Jaccard 相似度，衡量答案中关键词的覆盖率（阈值 0.3）
- **无答案处理**：标准答案为"无答案"时，预测正确得满分 1.0，否则得 0 分

评估结果保存至 `./data/metrics.json`，终端输出：

```
预测问题数：103, 预测最终得分：0.9357
```

---

## Docker 部署

```bash
# 构建镜像
docker build -t rag-with-chat .

# 运行容器（挂载模型目录避免重复下载）
docker run -it \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/pre_train_model:/app/pre_train_model \
  -v $(pwd)/data:/app/data \
  -e HF_TOKEN=your_token \
  --gpus all \
  rag-with-chat

# 自定义参数运行
docker run -it --gpus all rag-with-chat \
  python run.py --llm_name qwen2 --reranker_name bge
```

---

## 项目结构

```
rag_with_chat/
├── config.py                 # 全局配置（模型路径、设备设置）
├── run.py                    # 主入口：命令行参数解析 + 批量评估
├── example_test.py           # 交互式问答示例
├── generate_answer.py        # 核心调度：召回 + 重排 + Prompt 构建 + LLM 推理
├── hf_model.py               # 本地 HuggingFace LLM 推理模块
├── huggingface_proxy.py      # HuggingFace API 代理模块
├── rerank_model.py           # Cross-Encoder 重排模型
├── pdf_parse.py              # PDF 解析（多策略 + 多粒度分块）
├── test_score.py             # 评估指标计算
├── download_model.py         # 自动下载嵌入/重排模型
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 镜像配置
│
├── retriever/                # 四路检索器
│   ├── m3e_retriever.py      # M3E 稠密向量检索
│   ├── bge_retriever.py      # BGE 稠密向量检索
│   ├── bm25_retriever.py     # BM25 稀疏检索
│   └── tfidf_retriever.py    # TF-IDF 稀疏检索
│
├── data/                     # 数据目录
│   ├── car_user_manual.pdf   # 吉利汽车用户手册
│   ├── test_question.json    # 测试问题集
│   ├── gold_result.json      # 标准答案
│   └── result.json           # 预测结果（运行后生成）
│
├── models/                   # 本地 LLM 模型目录
│   ├── Qwen2-7B-Instruct/
│   ├── Baichuan2-7B-Chat/
│   └── chatglm3-6b/
│
├── pre_train_model/          # 嵌入 & 重排模型目录
│   ├── m3e-large/
│   ├── bge-m3/
│   ├── bge-reranker-large/
│   ├── bce-reranker-base_v1/
│   └── text2vec-base-chinese/
│
├── vector_db/                # 向量数据库缓存（运行后自动生成）
│   ├── faiss_m3e_index/
│   └── faiss_bge_index/
│
├── benchmark/                # 性能基准测试
├── ARCHITECTURE.md           # 系统架构设计文档
└── rag_architecture.html     # 可交互架构图
```

---

## 常见问题

**Q1: 运行时提示 `CUDA out of memory`**

降低 batch size 或使用较小的模型，也可切换到 API 模式：
```bash
python run.py --llm_name "Qwen/Qwen3.5-9B"
```

**Q2: 下载模型速度慢或失败**

国内用户推荐使用 ModelScope 镜像：
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 然后再运行 download_model.py
```

**Q3: ModuleNotFoundError: No module named 'dotenv'**

```bash
pip install python-dotenv
```

**Q4: 向量数据库重建很慢**

首次运行会构建 FAISS 向量索引，较为耗时（取决于 PDF 大小）。构建后会自动缓存到 `./vector_db/`，后续运行可通过 `--m3e_vector_path` 和 `--bge_vector_path` 参数复用缓存：
```bash
python run.py \
  --m3e_vector_path ./vector_db/faiss_m3e_index \
  --bge_vector_path ./vector_db/faiss_bge_index
```
（需要在 `run.py` 中取消对应参数的注释）



---

## License

本项目基于 [Apache License 2.0](LICENSE) 开源。
