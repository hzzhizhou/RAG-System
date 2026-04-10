"""语义感知分块：基于句子嵌入相似度动态切分"""
from typing import List, Optional
import numpy as np
from langchain_core.documents import Document
from config.settings import SEMANTIC_EMBEDDING_MODEL
from logs.log_config import data_layer_log as log
from .base import split_sentences, add_chunk_metadata

# 延迟加载嵌入模型
from sentence_transformers import SentenceTransformer
class SemanticChunker():
    """
    语义分块器
    原理：计算相邻句子的余弦相似度，相似度低于阈值处切分
    """
    _model = None
    def __init__(self, threshold: float = 0.6, buffer_size: int = 1, embedding_model:str = SEMANTIC_EMBEDDING_MODEL):
        """
        :param threshold: 相似度阈值，低于此值则切分（推荐 0.5~0.7）
        :param buffer_size: 切分时前后保留的句子数（上下文缓冲）
        """
        self.threshold = threshold
        self.buffer_size = buffer_size
        if SemanticChunker._model is None:
            SemanticChunker._model = SentenceTransformer(embedding_model)
        self.embed_model = SemanticChunker._model
        
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        log.info(f"开始编码 {len(sentences)} 个句子，预计需要较长时间...")
        embeddings = self.embed_model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
        log.info("编码完成")
        return embeddings

    def split_documents(self, docs: List[Document]) -> List[Document]:
        all_chunks = []

        for doc in docs:
            text = doc.page_content
            sentences = split_sentences(text)
            if len(sentences) <= 1:
                # 单句子或空文本直接作为一个块
                if sentences:
                    chunk = Document(page_content=sentences[0], metadata=doc.metadata.copy())
                    all_chunks.append(chunk)
                continue

            # 获取句子嵌入
            embeddings = self._get_sentence_embeddings(sentences)

            # 计算相邻句子相似度
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
                )
                similarities.append(sim)

            # 找出切分点（相似度低于阈值的位置）
            split_indices = [i+1 for i, sim in enumerate(similarities) if sim < self.threshold]

            # 构建块（带上下文缓冲）
            start = 0
            for idx in split_indices:
                actual_start = max(0, start - self.buffer_size)
                actual_end = min(len(sentences), idx + self.buffer_size)
                chunk_text = "".join(sentences[actual_start:actual_end])
                if chunk_text:
                    chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                    all_chunks.append(chunk)
                start = idx

            # 最后一块
            if start < len(sentences):
                actual_start = max(0, start - self.buffer_size)
                chunk_text = " ".join(sentences[actual_start:])
                if chunk_text:
                    chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                    all_chunks.append(chunk)

        all_chunks = add_chunk_metadata(all_chunks, chunk_type="semantic")
        log.info(f"语义分块完成，原始文档 {len(docs)} 个，生成 {len(all_chunks)} 个块")
        return all_chunks