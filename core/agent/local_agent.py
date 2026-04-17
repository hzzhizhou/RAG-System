# core/agent_layer.py
import asyncio
from typing import Any
from langchain.agents import create_agent # 使用新 API
from langchain_community.chat_models.tongyi import ChatTongyi
from core.agent.tools.retriever_tool import create_retriever_tool
import os
from dotenv import load_dotenv
load_dotenv()

def create_agent_with_memory(hybrid_retriever,llm=None):
    if llm is None:
        llm = ChatTongyi(
            model="qwen3-max",
            temperature=0.01,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            streaming=True
        )
    search_knowledge = create_retriever_tool(hybrid_retriever)
    tools = [search_knowledge]
    system_prompt = """
    你是一个智能助手，可以使用工具来回答问题。请按照以下格式逐步思考，思考过程要展现出来：

    Question: 用户的问题
    Thought: 我需要思考下一步该做什么
    Action: 需要采取的行动
    Action Input: 行动的输入参数
    Observation: 行动的结果
    ... (可以重复 Thought/Action/Action Input/Observation 多次)
    Thought: 我现在知道最终答案了
    Final Answer: 对用户问题的最终回答（包涵信息来源）

    注意：如果一次检索结果不够，可以多次调用 search_knowledge 工具，调整查询词。
    如果检索不到相关信息，请如实告知。

    开始！
    """

    # 使用新的 API 创建 Agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent