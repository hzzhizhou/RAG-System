"""分块器工厂：根据配置返回对应的分块器实例"""
from config.settings import (
    CHUNKING_STRATEGY,
    CHUNK_SIZE, CHUNK_OVERLAP,
    SEMANTIC_CHUNK_THRESHOLD, SEMANTIC_BUFFER_SIZE,
    SEMANTIC_EMBEDDING_MODEL,
    SLIDING_WINDOW_STEP,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
)
from logs.log_config import data_layer_log as log
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sliding_window import SlidingWindowChunker
from .parent_child import ParentChildChunker
from .combined_splitter import CombinedSplitter

def get_chunker(strategy: str = None):
    """
    获取分块器实例
    :param strategy: 可选值: recursive, semantic, sliding_window, parent_child
                     默认从 settings.CHUNKING_STRATEGY 读取
    :return: BaseChunker 子类实例
    """
    strategy = strategy or CHUNKING_STRATEGY
    log.info(f"初始化分块器，策略={strategy}")

    if strategy == "semantic":
        return SemanticChunker(
            threshold=SEMANTIC_CHUNK_THRESHOLD,
            buffer_size=SEMANTIC_BUFFER_SIZE,
            embedding_model=SEMANTIC_EMBEDDING_MODEL
        )
    elif strategy == "sliding_window":
        return SlidingWindowChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
    elif strategy == "parent_child":
        return ParentChildChunker(
            parent_chunk_size=PARENT_CHUNK_SIZE,
            parent_overlap=PARENT_CHUNK_OVERLAP,
            child_chunk_size=CHILD_CHUNK_SIZE,
            child_overlap=CHILD_CHUNK_OVERLAP
        )
    elif strategy == "combined_splitter":          # 组合分块策略，父块采用语义感知分块，子分块采用递归分块。
            return CombinedSplitter()
    else:  # 默认递归分块
        return RecursiveChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )