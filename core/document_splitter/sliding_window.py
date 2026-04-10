"""滑动窗口分块：固定大小、固定步长滑动（无自然边界）"""
from typing import List
from langchain_core.documents import Document
from logs.log_config import data_layer_log as log
from .base import add_chunk_metadata


class SlidingWindowChunker():
    """
    滑动窗口分块器
    严格按字符滑动，不考虑句子边界，适合需要极高召回的场景
    """

    def __init__(self, chunk_size: int, overlap: int):
        """
        :param chunk_size: 窗口大小（字符数）
        :param overlap: 重叠字符数（已废弃，使用 step 控制）
        :param step: 滑动步长，默认为 chunk_size - overlap
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap


    def split_documents(self, docs: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in docs:
            text = doc.page_content
            text_len = len(text)
            if text_len <= self.chunk_size:
                all_chunks.append(Document(page_content=text, metadata=doc.metadata.copy()))
                continue
            for start in range(0, text_len, self.step):
                end = min(start + self.chunk_size, text_len)
                chunk_text = text[start:end]
                all_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
        all_chunks = add_chunk_metadata(all_chunks, chunk_type="sliding_window")
        log.info(f"滑动窗口分块完成，生成 {len(all_chunks)} 个块（窗口大小={self.chunk_size}，步长={self.step}）")
        return all_chunks