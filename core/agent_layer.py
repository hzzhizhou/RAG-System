# core/agent_layer.py
import asyncio
from typing import Any
from langchain.agents import create_agent # 使用新 API
from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from core.retrieval.async_hybrid_retriever import HybridRetriever
from logs.log_config import log

def create_agent_with_memory(hybrid_retriever: HybridRetriever, llm: ChatTongyi) -> Any:
    """
    创建带记忆的 ReAct Agent（基于 LangChain 1.0+ 的 create_agent API）
    :param hybrid_retriever: 混合检索器实例
    :param llm: 大语言模型实例
    :return: Agent 实例（支持 ainvoke）
    """
    @tool
    def search_knowledge(query: str) -> str:
        """从知识库中检索与问题相关的信息。"""
        async def _async_search():
            docs = await hybrid_retriever._aget_relevant_documents(query)
            if not docs:
                return "未找到相关信息"
            contexts = []
            for i, doc in enumerate(docs[:3]):
                file_name = doc.metadata.get("file_name", "未知文档")
                content = doc.page_content[:500]
                contexts.append(f"【来源{i+1}：{file_name}】\n{content}")
            return "\n\n".join(contexts)
        try:
            return asyncio.run(_async_search())
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_async_search())
            raise

    tools = [search_knowledge]
    system_prompt = """
    你是一个智能助手，可以使用工具来回答问题。请根据对话历史逐步思考并给出最终答案。

    - 当你需要检索知识时，调用 search_knowledge 工具。
    - 回答必须基于检索到的内容，并在末尾标注信息来源（文档名）。
    - 如果检索不到相关信息，请如实告知。
    """

    # 使用新的 API 创建 Agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    log.info("Agent 创建成功，工具使用异步检索方法")
    return agent