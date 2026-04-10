from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from infrastructure.EmbeddingService.embedding_service import EmbeddingService
from typing import Any, List, Literal
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from infrastructure.vector_store.async_chroma_vector import ChromaVector
from utils.thread_pool_manager import init_thread_pools
from config.settings import BM25_WEIGHT, HYBRID_ASYNC_TIMEOUT, VECTOR_WEIGHT
from core.retrieval.bm25_retriever import Bm25Retriever
from core.retrieval.vector_retriever import VectorRetriever
from logs.log_config import retrieval_layer_log as log
import asyncio
import time
class HybridRetriever(BaseRetriever):
    _bm25_retriever: Bm25Retriever = PrivateAttr()
    _vector_retriever: VectorRetriever = PrivateAttr()
    _bm25_weight: float = PrivateAttr()
    _vector_weight: float = PrivateAttr()
    _fusion_method: str = PrivateAttr()
    _rrf_k: int = PrivateAttr()
    _embedding_service: Any = PrivateAttr()
    def __init__(self, 
                 bm25_retriever: Bm25Retriever, 
                 vector_retriever: VectorRetriever,
                 embedding_service,
                 rrf_k: int = 60,
                 **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_bm25_retriever', bm25_retriever)
        object.__setattr__(self, '_vector_retriever', vector_retriever)
        object.__setattr__(self, '_embedding_service', embedding_service)
        object.__setattr__(self, '_bm25_weight', BM25_WEIGHT)
        object.__setattr__(self, '_vector_weight', VECTOR_WEIGHT)
        object.__setattr__(self, '_rrf_k', rrf_k)

    def set_weights(self, weights: List[float]):
        if len(weights) != 2:
            raise ValueError("weights must contain exactly two values")
        self._bm25_weight, self._vector_weight = weights
        log.info(f"混合检索权重: BM25={self._bm25_weight}, Vector={self._vector_weight}")


    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        bm25_docs = self._bm25_retriever._get_relevant_documents(query, **kwargs) or "[]"
        vector_docs = self._vector_retriever._get_relevant_documents(query, **kwargs) or "[]"
        return self._fusion_rrf(bm25_docs, vector_docs)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        bm25_task = None
        vector_task = None
        embedding_task = None
        try:
            start=time.time()
           # 1. 同时启动 BM25 和 embedding 计算           
            bm25_task = asyncio.create_task(
                self._bm25_retriever.ainvoke(query, **kwargs)
            )
            start_time = time.time()
            embedding_task = asyncio.create_task(
                self._embedding_service.embed(query)
            )

            # 2. 等待 embedding 完成，然后启动向量检索
            query_vec = await embedding_task
            end_time = time.time() - start_time
            print(f"embeddings耗时: {end_time}s")
            vector_task = asyncio.create_task(
                self._vector_retriever.search_by_vector(query_vec, **kwargs)
            )
            bm25_docs, vector_docs = await asyncio.wait_for(
                asyncio.gather(bm25_task, vector_task),
                timeout=HYBRID_ASYNC_TIMEOUT
            )
            end =  time.time()-start
            log.info(f"混合检索完成,问题:{query} BM25: {len(bm25_docs)}条, Vector: {len(vector_docs)}条,耗时：{end}")
            return self._fusion_rrf(bm25_docs, vector_docs)
        except asyncio.TimeoutError:
            log.error(f"混合检索超时 (>{HYBRID_ASYNC_TIMEOUT}s) | 查询: {query[:50]}...")
            for task in (bm25_task, vector_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(
                *[t for t in (bm25_task, vector_task) if t and not t.done()],
                return_exceptions=True
            )
            raise
        except Exception as e:
            log.error(f"异步混合检索失败: {e}", exc_info=True)
            raise


    def _fusion_rrf(self, bm25_docs: List[Document], vector_docs: List[Document]) -> List[Document]:
        """RRF融合：移除嵌套，作为独立方法"""
        k = self._rrf_k
        doc_scores = {}
        doc_map = {}

        def get_key(doc: Document) -> str:
            if "id" in doc.metadata and doc.metadata["id"]:
                return f"id_{doc.metadata['id']}"
            source = doc.metadata.get("source", doc.metadata.get("file_path", ""))
            return f"{doc.page_content}_{source}"

        # 按RRF分数降序排序
        for rank, doc in enumerate(bm25_docs, start=1):
            key = get_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
            else:
                # 补充 bm25_score（如果 BM25 文档中携带）
                if "bm25_score" in doc.metadata:
                    doc_map[key].metadata["bm25_score"] = doc.metadata["bm25_score"]
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)

        # 处理向量文档
        for rank, doc in enumerate(vector_docs, start=1):
            key = get_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
            else:
                # 补充 vector_score 和 distance
                if "vector_score" in doc.metadata:
                    doc_map[key].metadata["vector_score"] = doc.metadata["vector_score"]
                if "distance" in doc.metadata:
                    doc_map[key].metadata["distance"] = doc.metadata["distance"]
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)

        # 排序返回
        merged = [doc_map[key] for key, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]
        return merged
