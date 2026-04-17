﻿﻿# RAG-System：企业级 RAG 智能问答系统

基于 LangChain + Chroma 构建的**生产级 RAG（检索增强生成）系统**，实现从文档加载、自适应分块、向量化存储到混合检索、重排序、流式生成、对话记忆、Agent 集成、RAGAS 评估的全链路解决方案。

##  系统架构

- **用户交互层**：Streamlit UI、FastAPI 端点
- **部署层**：API Key 鉴权、语义缓存、后台任务
- **生成层**：提示词模板、流式生成、上下文合并、对话存储
- **检索层**：智能路由、查询扩展、混合检索、RRF 融合、重排序
- **数据层**：文档加载、文本清洗、自适应分块、向量存储、嵌入服务
- **基础设施**：Chroma、BGE、Redis、通义千问

##  项目结构
    RAG-System/
    ├── config/                     # 配置模块（settings.py）
    ├── core/                       # 核心业务逻辑
    │   ├── chunking/               # 5种分块策略
    │   ├── retrieval/              # 检索器、重排序、路由、查询扩展
    │   ├── data_layer.py           # 数据入库
    │   ├── retriever_layer.py      # 检索服务
    │   ├── generation_layer.py     # 生成服务
    │   ├── evaluation_layer.py     # RAGAS 评估
    │   ├── deployment_layer.py     # FastAPI 部署
    │   ├── agent_layer.py          # Agent 集成
    │   └── chat_history_factory.py
    ├── infrastructure/             # 基础设施
    │   ├── vector_store/           # Chroma 异步封装
    │   ├── EmbeddingService/       # 嵌入服务（缓存+异步）
    │   └── redis/                  # Redis 对话历史
    ├── data_loader/                # 文档加载器（本地/网页）
    ├── utils/                      # 清洗、分词、安全、线程池
    ├── logs/                       # 日志目录
    ├── data/                       # 测试文档
    ├── main.py                     # 服务入口
    ├── streamlit_ui.py             # 演示界面
    └── requirements.txt

##  核心特性

- **自适应分块策略**：5种分块算法（递归/语义/滑动窗口/父子/组合），根据文档类型自动路由，**忠实度提升 15%**。
- **两阶段检索架构**：BM25 + 稠密向量混合检索 + RRF 融合 → Cross‑Encoder/Score 重排序，**上下文精度提升 115%**。
- **高性能本地嵌入**：集成 BAAI/bge‑base‑zh（768维），单例缓存 + 异步批处理，**Embedding 延迟降低 99%**（1.0s → 0.01s）。
- **全链路异步**：文档加载、分块、检索、生成均异步化，检索端到端耗时 **0.12s**，首字延迟 **<200ms**。
- **流式输出**：API 支持 SSE（Server‑Sent Events），前端逐字渲染，体验接近 ChatGPT。
- **对话记忆与改写**：基于 Redis 持久化历史，自动消除多轮对话中的指代歧义。
- **Agent 集成**：将 RAG 检索器封装为工具，ReAct Agent 可自主决策调用知识库。
- **自动化评估**：集成 RAGAS 框架，量化 faithfulness、answer_relevancy、context_precision/recall 等指标。

## 技术栈

- **框架**：LangChain, FastAPI, Streamlit
- **向量库**：Chroma（异步接口）
- **嵌入模型**：BAAI/bge-base-zh（本地），通义 text-embedding-v1（云端）
- **LLM**：通义千问
- **检索**：BM25 (rank_bm25)，RRF 融合，Cross‑Encoder (BAAI/bge-reranker-base)
- **对话存储**：Redis
- **评估**：RAGAS, datasets
- **异步**：asyncio, ThreadPoolExecutor

##  快速开始

### 环境要求
- Python 3.10+
- Langchain
- Chroma/Milvus
- Redis（可选，用于对话记忆）


### 安装与配置
```bash
git clone https://github.com/hzzhizhou/RAG-System.git
cd RAG-System
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 编辑 .env，填入你的 API_KEY
```
###  构建知识库
将文档放入 data/ 目录，然后执行：

```bash
python core/data_layer.py
```
###  启动API服务
```bash
python main.py
```
###  启动UI服务
```bash
streamlit run streamlit_ui.py
```
