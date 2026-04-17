# core/agent/tools/retriever_tool.py
from langchain_core.tools import tool
import asyncio

from core.retrieval.reranker import ScoreReranker
from core.retrieval.async_hybrid_retriever import HybridRetriever
from logs.log_config import log
def create_retriever_tool(hybrid_retriever:HybridRetriever):
    """工厂函数：创建知识库检索工具"""
    
    @tool
    def search_knowledge(query: str) -> str:
        """从知识库中检索与问题相关的信息。"""
        async def _async_search():
            log.info("从本地知识库中查询")
            results_list = await hybrid_retriever._aget_relevant_documents(query)
            reranker = ScoreReranker()
            final_docs =await reranker.rerank(results_list, query)
            if not final_docs:
                return "未找到相关信息"
            contexts = []
            for i, doc in enumerate(final_docs[:3]):
                file_name = doc.metadata.get("file_name", "未知文档")
                content = doc.page_content[:500]
                contexts.append(f"【来源{i+1}：{file_name}】\n{content}")
            log.info(f"{contexts}")
            return "\n\n".join(contexts)
        try:
            return asyncio.run(_async_search())
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_async_search())
            raise
    
    return search_knowledge