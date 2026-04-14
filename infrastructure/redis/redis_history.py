from datetime import datetime
from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage
from config.settings import REDIS_CONFIG
from infrastructure.redis.connection import get_redis_connection
from infrastructure.redis.base_chat_history import BaseChatHistory
import json
from logs.log_config import chat_history_log as log
class RedisChatHistory(BaseChatHistory):
    def __init__(self,session_id:str) -> None:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id 不能为空且必须为字符串")
        self.session_id  = session_id
        self.redis_client = get_redis_connection()
        self.key = f"{REDIS_CONFIG['key_prefix']}{self.session_id}"
        self.expire_seconds = REDIS_CONFIG["expire_days"] * 24 * 60 * 60
    
    def messages(self) -> List[BaseMessage]:
        """
        读取历史消息（转换为LangChain的Message对象）
        :return: 按时间排序的Message列表
        """
        try:
            # 读取Redis数据（无数据返回空列表）
            history_str = self.redis_client.get(self.key) or "[]"
            history_list = json.loads(history_str)
            
            # 转换为LangChain Message对象
            messages = []
            for item in history_list:
                # 兼容无type字段的旧数据
                msg_type = item.get("type", "human")
                content = item.get("content", "")
                if not content:
                    continue  # 跳过空内容消息
                
                if msg_type == "human":
                    messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    messages.append(AIMessage(content=content))
            
            log.debug(f"📖 会话[{self.session_id}] 读取历史消息{len(messages)}条")
            return messages
        except Exception as e :
            log.error(f"读取历史消息失败{e}")
    
    def add_message(self, message: BaseMessage):
        """
        添加消息到Redis（持久化）
        :param message: LangChain的HumanMessage/AIMessage对象
        """
        if not isinstance(message, (HumanMessage, AIMessage)):
            raise ValueError(f"仅支持HumanMessage/AIMessage，当前类型：{type(message)}")
        
        try:
            # 1. 获取现有历史
            history_str = self.redis_client.get(self.key) or "[]"
            history_list = json.loads(history_str)
            
            # 2. 构造新消息（添加时间戳，便于后续筛选）
            new_msg = {
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content.strip(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            history_list.append(new_msg)

            self.redis_client.set(
                self.key,
                json.dumps(history_list,ensure_ascii=False,indent = 2),
                ex = self.expire_seconds
            )
            log.info(f"会话[{self.session_id}],会话历史保存成功")
        except Exception as e:
            log.error(f'保存历史失败{e}')
            raise
    
    def clear(self):
        try:
            self.redis_client.delete(self.key)
            log.info(f"会话[{self.session_id}]会话历史已清空")
        except Exception as e:
            log.error(f'清空历史失败{e}')
            raise
    
    def rewrite_question(self, question: str, llm) -> str:
        """
        根据对话历史判断与用户的相关性，来决定是否改写问题，生成一个独立完整的问题。
        :param question: 用户当前问题
        :param llm: LangChain 的 LLM 实例（如 ChatTongyi）
        :return: 改写后的问题
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # 获取历史消息（最近 6 条，即 3 轮对话）
        history_messages = self.messages()[-6:] if self.messages() else []

        if not history_messages:
            # 无历史则直接返回原问题
            return question

        # 格式化历史
        history_text = ""
        for msg in history_messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            history_text += f"{role}：{msg.content}\n"

        # 构建改写提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询改写助手。可以判断对话历史与用户问题的相关性，来决定是否改写用户当前问题，改写后的问题确保不丢失任何关键信息。
            要求：
            1. 只输出改写后的问题
            2. 如果当前问题本身已经完整，可以不修改。
            3.多轮对话中，结合历史消除指代（如“它”、“那”）。
            4.修正错别字、口语化表达，使其更符合文档风格。
            5.补全省略的主语或条件。"""),
            ("human", "对话历史：\n{history}\n当前问题：{question}\n改写后的问题：")
        ])

        chain = prompt | llm | StrOutputParser()
        try:
            rewritten = chain.invoke({
                "history": history_text,
                "question": question
            }).strip()
            log.info(f"查询改写：原问题='{question}' → 改写后='{rewritten}'")
            return rewritten
        except Exception as e:
            log.error(f"查询改写失败，使用原问题：{e}")
            return question
