import time
from typing import Tuple
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import ROUTE_MODE, LLM_MODEL, LLM_TEMPERATURE, LLM_SEED
from logs.log_config import log

class SmartRouter:
    def __init__(self, bm25_retriever, vector_retriever, hybrid_retriever, llm):
        self.retrievers = {
            "bm25_retriever": bm25_retriever,
            "vector_retriever": vector_retriever,
            "hybrid_retriever": hybrid_retriever
        }
        self.llm = llm
        self.route_chain = self._init_route_chain()

    def _init_route_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你可以根据问题,返回以下标签之一：bm25/vector/hybrid
            - bm25_retriever:关键词明确的参数/定义查询（如"BM25 k1参数"）
            - vector_retriever:语义模糊的场景/原因查询（如"为什么向量检索更好"）
            - hybrid_retriever:通用查询（如"混合检索配置"）"""),
            ("human", "问题：{question}")
        ])
        return prompt | self.llm | StrOutputParser()

    async def route(self, question: str, mode: str = ROUTE_MODE) -> Tuple[BaseRetriever, str]:
        try:
            start_time = time.time()
            if mode == "rule":
                keyword_patterns = ["参数", "定义", "取值", "BM25", "分词"]
                semantic_patterns = ["为什么", "怎么", "如何", "场景", "对比"]
                if any(p in question for p in keyword_patterns):
                    retriever_type = "bm25_retriever"
                elif any(p in question for p in semantic_patterns):
                    retriever_type = "vector_retriever"
                else:
                    retriever_type = "hybrid_retriever"
            elif mode == "llm":
                result = await self.route_chain.ainvoke({"question": question})
                retriever_type = result.strip().lower()
                if retriever_type not in self.retrievers:
                    retriever_type = "hybrid_retriever"
            else:
                retriever_type = "hybrid_retriever"
            rout_time =  time.time()-start_time
            log.info(f"路由决策：{question[:50]}... → {retriever_type},路由耗时：{rout_time}")
            
            return self.retrievers[retriever_type], retriever_type
        except Exception as e:
            log.error(f"路由失败，降级为混合检索：{e}")
            return self.retrievers["hybrid_retriever"], "hybrid_retriever"