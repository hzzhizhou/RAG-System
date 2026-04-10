"""
企业级全局配置：集中管理所有参数，避免硬编码，便于环境切换
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量（企业级密钥管理）
load_dotenv()

# 项目路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"  # 原始文档目录
VECTOR_DB_DIR = BASE_DIR / "vector_db-768-combined"   # 向量库存储目录
LOG_DIR = BASE_DIR / "logs"              # 日志目录
BACKUP_DIR = BASE_DIR / "backups"        # 备份目录

# LLM/嵌入模型配置（企业级版本锁定）
LLM_MODEL = "qwen3-max"#换更快的模型：qwen-turbo 比 qwen3-max 快很多，效果差异不大。
LLM_TEMPERATURE = 0.01  # 无幻觉：固定温度
LLM_SEED = 42        # 固定随机种子（结果稳定）

# ====================== 嵌入模型配置 ======================
# ====================== 嵌入模型配置 ======================
EMBEDDING_BACKEND = "local"        # "local" 或 "dashscope"
# LOCAL_EMBEDDING_MODEL = "D:/RAG-Windows/AI大模型与智能体开发/models/bge-small-zh-v1.5"#---512维
LOCAL_EMBEDDING_MODEL = "D:/RAG-Windows/AI大模型与智能体开发/models/bge-base-zh-v1.5"#--768维
# 云端模型（仅当 EMBEDDING_BACKEND="dashscope" 时使用）
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v1"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 从.env读取

# 数据层配置
CHUNK_SIZE = 768               # 分块大小（适配LLM上下文）
CHUNK_OVERLAP = 128             # 块重叠（避免上下文丢失）
MAX_FILE_SIZE = 10 * 1024 * 1024  # 最大文件大小（10MB，规避内存溢出）
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx",".md"]  # 允许的文件格式


# ====================== 分块策略配置 ======================
CHUNKING_STRATEGY = "combined_splitter"  # 可选: recursive, semantic, sliding_window, parent_child,combined_splitter
# SEMANTIC_EMBEDDING_MODEL= "D:/RAG-Windows/AI大模型与智能体开发/models/bge-small-zh-v1.5"
SEMANTIC_EMBEDDING_MODEL = "D:/RAG-Windows/AI大模型与智能体开发/models/bge-base-zh-v1.5"#--768维
SEMANTIC_CHUNK_THRESHOLD = 0.6   # 句子相似度阈值（低于则切分）
SEMANTIC_BUFFER_SIZE = 1           # 切分时前后保留的句子数

# 滑动窗口分块参数
SLIDING_WINDOW_STEP = 256          # 滑动步长（若未设置，则使用 CHUNK_SIZE - CHUNK_OVERLAP）

# 父子分块参数
PARENT_CHUNK_SIZE = 2000           # 父块大小
PARENT_CHUNK_OVERLAP = 200         # 父块重叠
CHILD_CHUNK_SIZE = 400             # 子块大小
CHILD_CHUNK_OVERLAP = 40           # 子块重叠


# 检索层配置
RETRIEVER_K = 5                # 检索返回数量
BM25_WEIGHT = 0.4              # 混合检索BM25权重
BM25_SCORE_THRESHOLD = 0.1      #阙值过滤

VECTOR_WEIGHT = 0.6            # 混合检索向量权重
ROUTE_MODE = "rule"            # 路由模式：rule/llm/hybrid
RERANK_TOP_N = 3
RERANK_MODE = "score"


#查询扩展配置
USE_HYDE = False
USE_MULTI = False



# 评估层配置
EVAL_RUNS = 1                  # 评估次数（取平均，结果稳定）
EVAL_THRESHOLDS = {            # 企业级指标阈值（低于告警）
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.90,
    "context_recall": 0.95
}

# 部署层配置
API_HOST = "0.0.0.0"
API_PORT = 8000
CACHE_MAXSIZE = 1000           # 缓存最大条数
RESPONSE_TIMEOUT = 30          # API响应超时时间

# 日志配置
LOG_LEVEL = "INFO"             # 生产环境INFO，测试环境DEBUG
LOG_ROTATION = "100 MB"        # 日志轮转大小
LOG_RETENTION = "30 days"      # 日志保留时间（合规要求）
LOG_JSON_FORMAT = True         # 结构化日志（便于审计）
# 日志配置
LOG_CONSOLE_OUTPUT = True
LOG_FILE_OUTPUT = True

# Redis配置
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "password":os.getenv("Redis_password"),
    "db": 0,
    "key_prefix": "rag_chat_history:",  # 对话历史存储的key前缀
    "expire_days": 7,  # 过期天数
    "socket_timeout": 10,  # 连接超时时间
    "max_connections": 30  # 连接池最大连接数
}
#对话记忆层配置
SESSION_ID ="user_001"


# ---------------------------异步检索--------------------------------
# CPU核心数（自动适配）
import multiprocessing
CPU_CORES = multiprocessing.cpu_count()

# BM25检索（CPU密集）
# BM25_THREAD_POOL_SIZE 设置为 CPU 核心数的两倍，
# 适合 CPU 密集型检索（BM25 算法多线程并发），
# 通常这样可以充分利用多核资源，但避免线程过多导致频繁上下文切换。
BM25_THREAD_POOL_SIZE = CPU_CORES * 4
BM25_THREAD_POOL_NAME = "bm25-retriever-pool"
BM25_ASYNC_TIMEOUT = 10

# 向量检索（IO密集）
VECTOR_THREAD_POOL_SIZE = CPU_CORES * 8
VECTOR_THREAD_POOL_NAME = "vector-retriever-pool"
VECTOR_ASYNC_TIMEOUT = 10

# 混合检索（混合密集）
HYBRID_THREAD_POOL_SIZE = CPU_CORES * 4
HYBRID_THREAD_POOL_NAME = "hybrid-retriever-pool"
HYBRID_ASYNC_TIMEOUT = 10