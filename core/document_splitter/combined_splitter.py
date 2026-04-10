"""组合分块：语义分块生成父块，递归分块生成子块"""
import json
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from config.settings import (
    PARENT_CHUNK_SIZE,  CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    SEMANTIC_CHUNK_THRESHOLD, SEMANTIC_BUFFER_SIZE
)
from logs.log_config import data_layer_log as log
from .semantic import SemanticChunker
from .recursive import RecursiveChunker
from .base import add_chunk_metadata


class CombinedSplitter:
    """
    组合分块器：
    - 父块：使用语义感知分块（保持语义完整性）
    - 子块：对每个父块使用递归字符分块（固定大小+重叠）
    - 输出：子块列表（存入向量库），并生成父块映射（用于检索后获取完整上下文）
    """

    def __init__(self,
                 parent_threshold: float = SEMANTIC_CHUNK_THRESHOLD,
                 parent_buffer: int = SEMANTIC_BUFFER_SIZE,
                 parent_chunk_size: int = PARENT_CHUNK_SIZE,
                 child_chunk_size: int = CHILD_CHUNK_SIZE,
                 child_overlap: int = CHILD_CHUNK_OVERLAP):
        self.parent_chunker = SemanticChunker(
            threshold=parent_threshold,
            buffer_size=parent_buffer
        )
        self.child_splitter = RecursiveChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap
        )
        self.parent_chunk_size = parent_chunk_size

    def split_documents(self, docs: List[Document]) -> Tuple[List[Document], Dict[str, str]]:
        """
        对文档列表进行组合分块
        :return: (子块列表, 父块映射 {parent_id: parent_content})
        """
        parent_map = {}
        all_children = []
        parent_counter = 0
        for doc in docs:
            # 1. 语义分块生成父块
            parents = self.parent_chunker.split_documents([doc])
            # 2. 对每个父块递归分块生成子块
            for parent in parents:
                parent_id = f"{doc.metadata.get('file_name', 'unknown')}_p{parent_counter}"
                parent_counter += 1
                parent_content = parent.page_content
                parent_map[parent_id] = parent_content

                # 对父块内容递归分块
                children = self.child_splitter.split_documents([parent])
                
                for child in children:
                    # 子块继承父块的元数据，并添加 parent_id
                    child.metadata["parent_id"] = parent_id
                    child.metadata["parent_chunk_size"] = len(parent_content)
                    # 确保子块元数据中有原始文件信息
                    if "file_name" not in child.metadata:
                        child.metadata["file_name"] = doc.metadata.get("file_name", "unknown")
                    all_children.append(child)

        # 添加子块通用元数据
        all_children = add_chunk_metadata(all_children, chunk_type="child")
        log.info(f"组合分块完成：父块数={len(parent_map)}，子块数={len(all_children)}")
        return all_children, parent_map