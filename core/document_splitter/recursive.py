"""递归字符分块（原 DocumentSplitter 逻辑）"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from logs.log_config import data_layer_log as log
from .base import split_by_markdown_headers, add_chunk_metadata


class RecursiveChunker():
    """
    递归字符分块器（默认策略）
    先对 Markdown 文件按标题切分，再对超长块递归切分
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        # 1. 对 Markdown 文件进行结构切分
        structure_chunks = split_by_markdown_headers(docs)

        # 2. 递归控制长度
        final_chunks = []
        for chunk in structure_chunks:
            if len(chunk.page_content) > self.chunk_size:
                sub_chunks = self.text_splitter.split_documents([chunk])
                log.info(type(sub_chunks))
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # 3. 添加块级元数据
        final_chunks = add_chunk_metadata(final_chunks, chunk_type="recursive")
        log.info(f"递归分块完成，共 {len(final_chunks)} 块")
        return final_chunks