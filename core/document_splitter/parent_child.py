"""父子分块：保留大块作为父上下文，小块用于检索"""
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logs.log_config import data_layer_log as log
from .base import add_chunk_metadata


class ParentChildChunker():
    """
    父子分块器
    - 父块：较大块，保留完整上下文，不存入向量库（可单独存储）
    - 子块：小块，存入向量库用于检索，metadata 中记录 parent_id
    检索时可通过 parent_id 获取父块内容提供更丰富上下文
    """

    def __init__(self, parent_chunk_size: int, parent_overlap: int,
                 child_chunk_size: int, child_overlap: int):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        # 存储父块映射（可选，若需在检索后获取父块内容可持久化）
        self.parent_map: Dict[str, Document] = {}

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        返回子块列表（用于向量存储），同时内部记录父块
        """
        all_children = []
        parent_counter = 0

        for doc in docs:
            # 1. 生成父块
            parents = self.parent_splitter.split_documents([doc])
            for parent in parents:
                parent_id = f"{doc.metadata.get('file_name', 'unknown')}_p{parent_counter}"
                parent.metadata["parent_id"] = parent_id
                parent.metadata["chunk_type"] = "parent"
                self.parent_map[parent_id] = parent
                parent_counter += 1

                # 2. 对每个父块生成子块
                children = self.child_splitter.split_documents([parent])
                for child in children:
                    child.metadata["parent_id"] = parent_id
                    child.metadata["chunk_type"] = "child"
                    all_children.append(child)

        all_children = add_chunk_metadata(all_children, chunk_type="child")
        log.info(f"父子分块完成：父块数={len(self.parent_map)}，子块数={len(all_children)}")
        return all_children

    def get_parent(self, parent_id: str) -> Document:
        """根据 parent_id 获取父块文档"""
        return self.parent_map.get(parent_id)

    def get_parents_batch(self, parent_ids: List[str]) -> List[Document]:
        """批量获取父块"""
        return [self.parent_map[pid] for pid in parent_ids if pid in self.parent_map]