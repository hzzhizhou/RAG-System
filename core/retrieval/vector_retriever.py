from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.thread_pool_manager import init_thread_pools
from infrastructure.EmbeddingService.embedding_service import EmbeddingService
sys.path.append(str(Path(__file__).parent.parent.parent))
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from config.settings import RETRIEVER_K
from logs.log_config import retrieval_layer_log as log
import asyncio
import time

class VectorRetriever(BaseRetriever):
    vector_store: Any = Field(description="向量库实例")
    k: int = Field(default=RETRIEVER_K, description="检索返回条数")
    # 新增：embedding_service 不是 Pydantic 字段，用 PrivateAttr
    _embedding_service: Any = None
    def __init__(self, vector_store, embedding_service,k: int = RETRIEVER_K):
        super().__init__(vector_store=vector_store, k=k)
        # 父类已设置 self.vector_store，无需重复赋值
        object.__setattr__(self, '_embedding_service', embedding_service)
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        try:
            start_time = time.time()
            filter_condition = kwargs.get('filter')
            results = self.vector_store.similarity_search_with_score(
                query, k=self.k, filter=filter_condition
            )
            docs = []
            for doc_obj, score in results:
                # 企业级：将距离转换为 [0,1] 相似度（倒数映射，适用于 L2/余弦距离）
                similarity = 1.0 / (1.0 + score)
                doc_obj.metadata["vector_score"] = similarity
                docs.append(doc_obj)
            docs.sort(key=lambda x: x.metadata.get("vector_score", 0), reverse=True)
            log.info(f"向量检索完成,问题:{query},返回{len(docs)}个文档，耗时{time.time()-start_time:.3f}s")
            return docs
        except Exception as e:
            log.error(f"向量检索失败: {e}", exc_info=True)
            raise
    
    async def search_by_vector(self, vector: List[float], **kwargs) -> List[Document]:
        """直接使用预计算向量进行检索（异步）"""
        start_time = time.time()
        filter_condition = kwargs.get('filter')
        results = await self.vector_store.asimilarity_search_by_vector(
            vector, k=self.k, filter=filter_condition
        )
        docs = []
        for doc, distance in results:
            similarity = 1.0 / (1.0 + distance)
            doc.metadata["distance"] = distance
            doc.metadata["vector_score"] = similarity
            docs.append(doc)
        docs.sort(key=lambda x: x.metadata.get("vector_score", 0), reverse=True)
        log.info(f"向量检索完成，返回{len(docs)}个文档，耗时{time.time()-start_time:.3f}s")
        return docs
    
    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """异步检索（内部使用 embedding_service 获取向量）"""
        # 获取 embedding（自动缓存）
        start_time = time.time()
        query_vec = await self._embedding_service.embed(query)
        # 使用向量检索
        end_time = time.time() - start_time
        print(f"embeddings耗时: {end_time}s")
        return await self.search_by_vector(query_vec, **kwargs)
    
if __name__ =='__main__':
    vector_store = ChromaVector()
    embedding_service = EmbeddingService(vector_store.embedding_model)
    vector_retriever = VectorRetriever(vector_store,embedding_service)
    start = time.time()
    async def test():
        tasks = [vector_retriever.ainvoke(q) for q in questions]
        results = await asyncio.gather(*tasks)
        return results
    questions =[ 
        "什么是RAG",
        "分块大小怎么确定",
        "P99响应时间多少合适"
    ]

    asyncio.run(test())
    end= time.time()-start
    print(f"耗时:{end}")


