"""
检索层:Bm25检索,相似度检索,混合检索
通过路由实现智能检索，选择合适的检索方式
实现HyDE,重排序
"""
from functools import partial
from infrastructure.EmbeddingService.embedding_service import EmbeddingService
from typing import List, Tuple
from langchain_core.documents import Document
from config.settings import ROUTE_MODE
from logs.log_config import retrieval_layer_log as log
from core.retrieval.bm25_retriever import Bm25Retriever
from core.retrieval.vector_retriever import VectorRetriever
from core.retrieval.async_hybrid_retriever import HybridRetriever
from core.retrieval.router import SmartRouter
from core.retrieval.reranker import ScoreReranker
from core.retrieval.query_enhancer import QueryEnhancer
import time
import asyncio

class RetrievalService:
    def __init__(self, vector_store, llm, chat_history):
        self.llm = llm
        self.vector_store = vector_store
        self.chat_history = chat_history
        self.embedding_service = EmbeddingService(vector_store.embedding_model)
        self.bm25_retriever = Bm25Retriever(self.vector_store)
        self.vector_retriever = VectorRetriever(self.vector_store,self.embedding_service)
        self.hybrid_retriever = HybridRetriever(self.bm25_retriever, self.vector_retriever,self.embedding_service)
        self.router = SmartRouter(self.bm25_retriever, self.vector_retriever, self.hybrid_retriever, self.llm)
        self.reranker = ScoreReranker()
        self.query_enhancer = QueryEnhancer(self.llm)

    async def _merge_results(self, docs: List[Document], original_question: str) -> List[Document]:
        return await self.reranker.rerank(docs,original_question)

    async def retrieve(self, question: str, route_mode: str = ROUTE_MODE,
                 use_context: bool = True,
                 use_hyde: bool = False,
                 use_multi: bool = False,
                 session_id: str = None) -> Tuple[List[Document], str]:
        start_time = time.time()
        original_question = question

        # 1. 预取 embedding（基于原始问题）
        prefetch_task = asyncio.create_task(self.embedding_service.embed(original_question))

        # 2. 异步执行查询改写
        if use_context and self.chat_history:
            loop = asyncio.get_running_loop()
            rewrite_func = partial(self.chat_history.rewrite_question, original_question, self.llm)
            rewritten_question = await loop.run_in_executor(None, rewrite_func)
            if rewritten_question != original_question:
                log.info(f"查询改写: {original_question} -> {rewritten_question}")
        else:
            rewritten_question = original_question

        # 3. 路由决策（使用改写后的问题）
        retriever, retriever_type = await self.router.route(rewritten_question, route_mode)

        # 4. 多查询/HyDE（使用改写后的问题）
        queries = [rewritten_question]
        if use_multi:
            queries = await self.query_enhancer.multi_query(rewritten_question)
            log.info(f"生成的多个问题{queries}")
        elif use_hyde:
            queries = [await self.query_enhancer.hyde(rewritten_question)]
        
        ret_time = time.time() - start_time
        print(f"查询改写耗时:{ret_time}")
        retriever_time = time.time()
        all_docs = []
        results_list = await asyncio.gather(*[retriever.ainvoke(q) for q in queries])
        all_docs = [doc for sublist in results_list for doc in sublist]
        final_docs =await self._merge_results(all_docs, question)
        end_time = time.time()-retriever_time
        log.info(f"检索完成，检索器={retriever_type}，返回文档数={len(final_docs)},检索总耗时：{end_time:.2f}秒")
        return final_docs, retriever_type