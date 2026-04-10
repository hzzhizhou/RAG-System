# core/agent_layer.py
import asyncio
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from core.retrieval.async_hybrid_retriever import HybridRetriever
from logs.log_config import log

def create_agent_with_memory(hybrid_retriever: HybridRetriever, llm: ChatTongyi) -> AgentExecutor:
    """
    创建带记忆的 ReAct Agent
    :param hybrid_retriever: 混合检索器实例（用于知识检索）
    :param llm: 大语言模型实例
    :return: AgentExecutor 实例
    """

    @tool
    def search_knowledge(query: str) -> str:
        """
        从知识库中检索与问题相关的信息。
        :param query: 用户的查询问题
        :return: 检索到的文档内容（最多3个文档，每个最多500字符）
        """
        # 由于 AgentExecutor 运行在独立线程中（通过 asyncio.to_thread），
        # 该线程没有运行中的事件循环，可以安全使用 asyncio.run()
        async def _async_search():
            # 调用异步检索方法
            docs = await hybrid_retriever._aget_relevant_documents(query)
            if not docs:
                return "未找到相关信息"
            # 取前3个文档，每个最多500字符
            contexts = []
            for i, doc in enumerate(docs[:3]):
                file_name = doc.metadata.get("file_name", "未知文档")
                content = doc.page_content[:500]
                contexts.append(f"【来源{i+1}：{file_name}】\n{content}")
            return "\n\n".join(contexts)
        try:
            return asyncio.run(_async_search())
        except RuntimeError as e:
            # 如果已有事件循环（例如在异步环境中直接调用），则使用当前循环
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_async_search())
            raise

    tools = [search_knowledge]

    # ReAct 提示词模板，支持对话历史
    template = """你是一个智能助手，可以使用工具来回答问题。请根据以下对话历史和用户问题，逐步思考并给出最终答案。

对话历史：
{chat_history}

用户问题：{input}

你可以使用以下工具：
{tools}

请使用以下格式：
Thought: 你需要思考下一步该做什么
Action: 需要采取的行动，必须是 [{tool_names}] 之一
Action Input: 行动的输入参数
Observation: 行动的结果
... (可以重复 Thought/Action/Action Input/Observation 多次)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答(包涵回答来源)

开始！

{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        return_intermediate_steps=True
    )
    log.info("Agent 执行器创建成功，工具使用异步检索方法")
    return agent_executor