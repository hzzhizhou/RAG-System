"""分块公共工具函数"""
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from logs.log_config import data_layer_log as log

# Markdown 标题层级配置
HEADERS_TO_SPLIT_ON = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]


def split_by_markdown_headers(docs: List[Document]) -> List[Document]:
    """
    对文档列表按 Markdown 标题结构进行预切分（保留原始元数据）
    :param docs: 原始文档列表
    :return: 按标题切分后的文档列表（未进行大小控制）
    """
    result = []
    try:
        md_splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT_ON)
        for doc in docs:
            # 仅对 markdown 文件进行结构切分
            if doc.metadata.get("file_type") == "md":
                chunks = md_splitter.split_text(doc.page_content)
                # 将原始元数据复制到每个子块
                for chunk in chunks:
                    chunk.metadata.update(doc.metadata.copy())
                result.extend(chunks)
            else:
                result.append(doc)
    except Exception as e:
        log.error(f"Markdown 结构切分失败: {e}")
        return docs
    return result


def add_chunk_metadata(chunks: List[Document], chunk_type: str = "normal") -> List[Document]:
    """
    为分块添加标准元数据（chunk_id, chunk_index, chunk_type）
    :param chunks: 分块列表
    :param chunk_type: 块类型标识（normal, child, parent 等）
    :return: 添加元数据后的列表
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i+1}"
        chunk.metadata["chunk_index"] = i
        if "chunk_type" not in chunk.metadata:
            chunk.metadata["chunk_type"] = chunk_type
    return chunks


def split_sentences(text: str) -> List[str]:
    """
    简单的中英文句子分割（基于标点）
    :param text: 原始文本
    :return: 句子列表
    """
    # 匹配句号、感叹号、问号、分号等结尾的句子
    pattern = r'(?<=[。！？!?;；])\s*'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]