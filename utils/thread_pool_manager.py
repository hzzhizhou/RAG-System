from concurrent.futures import ThreadPoolExecutor
from config.settings import (
    BM25_THREAD_POOL_SIZE, BM25_THREAD_POOL_NAME,
    VECTOR_THREAD_POOL_SIZE, VECTOR_THREAD_POOL_NAME,
    HYBRID_THREAD_POOL_SIZE, HYBRID_THREAD_POOL_NAME
)
from logs.log_config import log

# 全局线程池字典（单例，避免重复创建）
THREAD_POOLS = {}

def init_thread_pools():
    """服务启动时初始化所有检索线程池"""
    try:
        # BM25线程池
        THREAD_POOLS["bm25"] = ThreadPoolExecutor(
            max_workers=BM25_THREAD_POOL_SIZE,
            thread_name_prefix=BM25_THREAD_POOL_NAME
        )
        # 向量检索线程池
        THREAD_POOLS["vector"] = ThreadPoolExecutor(
            max_workers=VECTOR_THREAD_POOL_SIZE,
            thread_name_prefix=VECTOR_THREAD_POOL_NAME
        )
        # 混合检索线程池
        THREAD_POOLS["hybrid"] = ThreadPoolExecutor(
            max_workers=HYBRID_THREAD_POOL_SIZE,
            thread_name_prefix=HYBRID_THREAD_POOL_NAME
        )
        log.info("所有检索线程池初始化完成")
    except Exception as e:
        log.error(f"线程池初始化失败：{e}")
        raise

def get_thread_pool(pool_type: str) -> ThreadPoolExecutor:
    """获取指定类型的线程池"""
    if pool_type not in THREAD_POOLS:
        raise ValueError(f"未知的线程池类型：{pool_type}")
    return THREAD_POOLS[pool_type]

def shutdown_thread_pools():
    """服务优雅关闭时销毁线程池"""
    for pool_type, pool in THREAD_POOLS.items():
        pool.shutdown(wait=True)
        log.info(f"{pool_type}线程池已关闭")