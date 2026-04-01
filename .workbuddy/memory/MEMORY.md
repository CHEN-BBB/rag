# MEMORY.md — 长期记忆

## 项目：rag_with_chat

**路径**：`c:/Users/86138/Desktop/all/CODE/rag_with_chat`  
**定位**：基于吉利汽车用户手册的 RAG 知识问答系统

### 核心技术架构
- 四路召回：M3E（稠密）+ BGE（稠密）+ BM25（稀疏）+ TF-IDF（稀疏）
- 两阶段检索：Bi-Encoder 召回 → Cross-Encoder 精排（BCE/BGE Reranker）
- 双 LLM 路径：本地 HF（hf_model.py）/ 云端 API（huggingface_proxy.py）
- 支持 LLM：Qwen2-7B / Baichuan2-7B / ChatGLM3-6B / Qwen3-9B(API)

### 已知 Bug
- `retriever/bm25_retriever.py` 第55行有 `break`，导致 BM25 只返回1条文档

### 生成的文档
- `rag_architecture.html`：可交互系统架构图（4个标签页）
- `ARCHITECTURE.md`：完整设计文档（8章节）
