import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from core.agent.tools.webpage_fetcher import fetch_webpage
from core.agent.tools.web_search import web_search

load_dotenv()

def create_web_agent(llm=None):
    if llm is None:
        llm = ChatTongyi(
            model="qwen3-max",
            temperature=0.01,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            streaming=True
        )
    tools = [fetch_webpage, web_search]
    web_prompt = """
    你是一个网络信息助手，必须使用工具来回答问题。请严格按照以下 ReAct 格式逐步思考，思考过程要展现出来：

    Question: 用户的问题
    Thought: 我需要思考下一步该做什么
    Action: 需要采取的行动，必须是 [web_search, fetch_webpage] 之一
    Action Input: 行动的输入参数
    Observation: 行动的结果
    ... (可以重复 Thought/Action/Action Input/Observation 多次)
    Thought: 我现在知道最终答案了
    Final Answer: 对用户问题的最终回答（必须注明信息来源）

    规则：
    1. 如果用户提供具体网址，必须使用 fetch_webpage 抓取内容。
    2. 对于任何需要实时信息、新闻、事实查询的问题，必须使用 web_search 工具。
    3. 如果 web_search 返回的摘要不够详细，可以再次调用 web_search 使用不同关键词，或者对搜索结果中的某个链接调用 fetch_webpage 获取完整内容。
    4. 严禁直接编造答案，所有信息必须来自工具返回的结果。
    5. 最终回答必须注明信息来源（网址或文档名）。

    开始！
    """
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=web_prompt
    )
    return agent