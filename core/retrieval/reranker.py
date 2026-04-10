from typing import List
from langchain_core.documents import Document
from config.settings import RERANK_TOP_N, BM25_WEIGHT, VECTOR_WEIGHT
from logs.log_config import retrieval_layer_log as log
import time
import asyncio
# core/retrieval/reranker.py

class CrossEncoderReranker:
    """使用 Cross-Encoder 进行重排序（轻量级、高性能）"""
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = RERANK_TOP_N):
        self.top_n = top_n
        self.model_name = model_name
        self._model = None  # 延迟加载

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            log.info(f"Cross-Encoder 模型加载完成: {self.model_name}")

    async def rerank(self, docs: List[Document], query: str) -> List[Document]:
        """异步重排序，返回 top_n 个文档"""
        if len(docs) <= 1:
            return docs[:self.top_n]
        start_time = time.time()
        self._load_model()
        pairs = [(query, doc.page_content) for doc in docs]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._model.predict, pairs)
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)
        # 按分数降序排序
        docs.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        reranked = docs[:self.top_n]
        log.info(f"Cross-Encoder 重排序完成，返回 {len(reranked)} 个文档，耗时 {time.time()-start_time:.3f}s")
        return reranked


class ScoreReranker:
    """
    基于已有分数的重排序（无需外部模型）
    直接使用文档中已存储的 bm25_score 和 vector_score 进行加权融合
    """
    def __init__(self, top_n: int = RERANK_TOP_N):
        self.top_n = top_n

    async def rerank(self, docs: List[Document], query: str = "") -> List[Document]:
        """
        异步重排序接口（保持与 CrossEncoderReranker 一致）
        """
        start_time = time.time()
        if len(docs) <= 1:
            return docs[:self.top_n]

        # 提取分数，如果文档中已经存在归一化后的分数则优先使用，否则使用原始分数并自行归一化
        bm25_scores = []
        vector_scores = []
        for doc in docs:
            # 优先使用已经归一化好的分数（如果有）
            norm_bm25 = doc.metadata.get("norm_bm25_score")
            norm_vector = doc.metadata.get("norm_vector_score")
            if norm_bm25 is not None and norm_vector is not None:
                bm25_scores.append(norm_bm25)
                vector_scores.append(norm_vector)
            else:
                # 降级：使用原始分数并实时归一化
                raw_bm25 = doc.metadata.get("bm25_score", 0.0)
                raw_vector = doc.metadata.get("vector_score", 0.0)
                bm25_scores.append(raw_bm25)
                vector_scores.append(raw_vector)

        # 如果所有文档都没有分数（理论上不应发生），则返回原顺序
        if not bm25_scores:
            return docs[:self.top_n]

        # 实时归一化（Min-Max，仅在当前批次内）
        def normalize(scores):
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return [0.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        norm_bm25 = normalize(bm25_scores)
        norm_vector = normalize(vector_scores)

        # 计算加权融合分数
        fusion_scores = [
            BM25_WEIGHT * b + VECTOR_WEIGHT * v
            for b, v in zip(norm_bm25, norm_vector)
        ]

        # 按融合分数降序排序
        for i, doc in enumerate(docs):
            doc.metadata["fusion_score"] = fusion_scores[i]

        reranked = sorted(docs, key=lambda x: x.metadata.get("fusion_score", 0), reverse=True)
        reranked = reranked[:self.top_n]

        log.info(f"分数重排序完成，返回 {len(reranked)} 个文档，耗时 {time.time()-start_time:.3f}s")
        # for chunk in reranked:
        #     print(chunk.metadata['fusion_score'])
        return reranked