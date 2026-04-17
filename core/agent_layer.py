import os
import asyncio
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from dotenv import load_dotenv
from core.retrieval.async_hybrid_retriever import HybridRetriever
from core.agent.tools.web_search import web_search
from core.agent.tools.webpage_fetcher import fetch_webpage

load_dotenv()

def create_unified_agent(hybrid_retriever: HybridRetriever, llm=None):
    """
    创建全能 Agent，拥有三个工具：
    - search_knowledge: 内部知识库检索
    - web_search: 互联网搜索
    - fetch_webpage: 抓取指定网页内容
    """
    if llm is None:
        llm = ChatTongyi(
            model=os.getenv("LLM_MODEL", "qwen3-max"),
            temperature=0.01,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            streaming=True
        )

    # 定义内部知识库检索工具（需要注入 hybrid_retriever）
    @tool
    def search_knowledge(query: str) -> str:
        """从企业内部知识库中检索信息，适用于公司文档、技术手册、FAQ、历史记录等。"""
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
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_async_search())

    tools = [search_knowledge, web_search, fetch_webpage]
    system_prompt = """
    你是一个全能智能助手，可以使用以下三个工具来回答问题：
    1. **search_knowledge**：查询企业内部知识库（公司文档、技术规范、内部 FAQ）。适用于关于公司内部产品、技术细节、流程的问题。
    2. **web_search**：搜索互联网获取实时信息。适用于新闻、最新动态、外部知识、用户未提供网址的查询。
    3. **fetch_webpage**：抓取用户提供的具体网址的详细内容。
    ## 决策规则：
    - 如果问题明确涉及公司内部信息（如“我们的RAG系统如何配置”），使用 search_knowledge。
    - 如果问题需要最新新闻、实时数据、外部信息（如“今天有什么AI新闻”），使用 web_search。
    - 如果用户提供了具体网址（如“总结这个网页：https://...”），使用 fetch_webpage。
    - 对于复杂问题（如“对比我们公司的RAG方案和最新的开源RAG技术”），可以**先调用 search_knowledge，再调用 web_search**，然后综合两者给出答案。
    - 如果第一次检索结果不理想，可以调整查询词再次调用同一工具。

    ## 回答要求：
    - 所有信息必须来自工具返回的结果，严禁编造。
    - 回答末尾必须注明信息来源（文档名或网址）。
    - 如果无法找到相关信息，请如实告知。

    ## ReAct 格式示例：
    Question: 用户的问题
    Thought: 我需要思考下一步该做什么
    Action: 需要采取的行动（工具名）
    Action Input: 工具的输入参数
    Observation: 工具返回的结果
    ...（可以重复 Thought/Action/Observation 多次）
    Thought: 我现在知道最终答案了
    Final Answer: 对用户问题的最终回答（注明来源）

    开始！
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent