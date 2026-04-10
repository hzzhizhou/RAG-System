#--------------------第二步:文档分块----------------------

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CHUNK_OVERLAP, CHUNK_SIZE

from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from logs.log_config import data_layer_log as log
headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]
class DocumentSplitter:
    @staticmethod
    def split_documents(docs: List[Document]) -> List[Document]:
        # 1. 按结构粗切分
        structure_chunks = []
        try:
            for doc in docs:
                # 根据文档类型调用不同的解析器（可扩展）
                if doc.metadata.get("file_type") == "md":
                    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
                    structure_chunks.extend(md_splitter.split_text(doc.page_content))
                else:
                    # 使用递归分块器
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
                    )
                    structure_chunks.extend(text_splitter.split_documents([doc]))
        except Exception as e:
            log.error(f"文档分块失败:{e}")
            raise
        # 2. 长度控制与递归精切分
        final_chunks = []
        for chunk in structure_chunks:
            if len(chunk.page_content) > CHUNK_SIZE:
                sub_chunks = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["。", "！", "？", "；",  " ", ""]
                ).split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        # 3. 添加块级元数据
        for i, chunk in enumerate(final_chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i+1}"
            chunk.metadata["chunk_index"] = i
        log.info(f"文档分块成功，总共分了{len(final_chunks)}块")
        return final_chunks