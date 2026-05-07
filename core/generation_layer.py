"""
生成层：无幻觉生成+提示词工程+格式标准化+复杂问题处理
通过AnswerGenerator类，可以生成答案，支持流式生成和非流式生成
"""
from typing import List, Optional, AsyncIterator
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.context_format import format_context_with_parents
from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_SEED, DASHSCOPE_API_KEY
)
from logs.log_config import generation_layer_log as log
from core.chat_history_factory import init_chat_history
from langchain_core.messages import HumanMessage, AIMessage
import time

class AnswerGenerator:
    def __init__(self, llm: Optional[ChatTongyi] = None):
        self.llm = llm or ChatTongyi(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
            api_key=DASHSCOPE_API_KEY,
            streaming=True
        )
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能回答助手，严格按照以下要求回答用户问题：
            要求：
                1. 回答简洁准确,核心内容不超过100字;
                2. 优先使用【对话历史】中的信息回答关于对话上下文的问题（如“上一个问题是什么”）。
                3. 仅使用资料中的知识回答，不添加额外信息。
                4. 必须在回答末尾标注信息来源（格式：「来源：文档名」，若来自对话历史则标注「来源：对话历史」）；
                5. 无相关资料时，仅回复「无相关信息」。
            资料：{context}
            """),
            ("human", "用户问题：{question}")
        ])
        self.answer_chain = self.base_prompt | self.llm | StrOutputParser()
        self.parent_cache = {}
        self._load_parent_cache()
    
    def _load_parent_cache(self):
        """加载父块映射文件（如果存在）"""
        cache_path = Path(__file__).parent.parent / "parent_cache.json"
        if cache_path.exists():
            import json
            with open(cache_path, "r", encoding="utf-8") as f:
                self.parent_cache = json.load(f)
            log.info(f"加载父块映射，共 {len(self.parent_cache)} 个父块")
        else:
            log.warning("未找到 parent_cache.json，父子分块功能不可用")

    def format_context(self, docs: List[Document]) -> str:
        return format_context_with_parents(
            docs=docs,
            parent_cache=self.parent_cache,
            max_context_length=4000
        )

    async def stream_generate(self, question: str, docs: List[Document], session_id: Optional[str] = None) -> AsyncIterator[str]:
        """
        流式生成答案，逐步产出 token。
        注意：不存储对话历史（因为无法获得完整答案），如需存储请在外部收集完整答案后调用存储逻辑。
        """
        start_time = time.time()
        context_str = self.format_context(docs)
        # 添加对话历史
        if session_id:
            chat_hist = init_chat_history(session_id)
            messages = chat_hist.messages()  # 获取历史消息列表
            if messages:
                history_lines = []
                for msg in messages[-2:]:  # 最近2条消息（1轮对话）
                    role = "User" if isinstance(msg, HumanMessage) else "AI"
                    history_lines.append(f"{role}：{msg.content}")
                if history_lines:
                    history_str = "\n".join(history_lines)
                    # 将对话历史拼接到上下文前面
                    context_str = f"对话历史:\n{history_str}\n\n上下文:\n{context_str}"
                    print(context_str)
        try:
            async for chunk in self.answer_chain.astream({
                "question": question,
                "context": context_str
            }):
                yield chunk
            end_time = time.time()
            log.info(f"回答生成完成 | 问题：{question[:30]},生成耗时：{end_time - start_time:.2f}秒")
        except Exception as e:
            log.error(f"回答生成失败：{str(e)}")
            yield "系统异常，无法生成回答"

    async def generate(self, question: str, docs: List[Document], session_id: Optional[str] = None) -> str:
        """非流式生成（保留原接口，内部使用流式拼接）"""
        full_answer = []
        async for chunk in self.stream_generate(question, docs, session_id):
            full_answer.append(chunk)
        answer = "".join(full_answer)
        # 存储对话历史
        if session_id:
            chat_history = init_chat_history(session_id)
            chat_history.add_message(HumanMessage(content=question))
            chat_history.add_message(AIMessage(content=answer))
        print(f"AI回答:\n{answer}")
        return answer