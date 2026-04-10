"""
生成层：无幻觉生成+提示词工程+格式标准化+复杂问题处理
企业级优化：单LLM实例复用，差异化参数控制
核心修改：来源标注从chunk_id改为文档名称（file_name）
"""
from typing import List, Optional, AsyncIterator
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.thread_pool_manager import init_thread_pools
from core.retriever_layer import RetrievalService
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_SEED, DASHSCOPE_API_KEY, SESSION_ID
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
                1. 回答简洁准确，核心内容不超过200字；
                2. 仅使用提供的资料回答，不添加任何额外信息、推测或编造内容；
                3. 必须在回答末尾标注信息来源（格式：「来源：文档名」，多个来源用逗号分隔）；
                4. 无相关资料时，仅回复「无相关信息」。
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
        if not docs:
            return "无参考资料"
        # 1. 分离父子分块的子块和普通文档
        parent_scores = {}  # {parent_id: {"score": float, "doc": Document (任意子块)}}
        normal_docs = []
        
        for doc in docs:
            # 优先使用 fusion_score（重排序后的分数），其次使用 rerank_score，最后使用 vector_score
            score = doc.metadata.get("fusion_score") or doc.metadata.get("rerank_score") or doc.metadata.get("vector_score", 0.0)
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_cache:
                if parent_id not in parent_scores:
                    parent_scores[parent_id] = {"score": score, "doc": doc}
                else:
                    # 取最高分（也可取平均分）
                    parent_scores[parent_id]["score"] = max(parent_scores[parent_id]["score"], score)
            else:
                normal_docs.append(doc)
        
        # 2. 按父块分数排序
        sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # 3. 构建上下文
        context_parts = []
        for parent_id, info in sorted_parents:
            parent_content = self.parent_cache[parent_id]
            file_name = info["doc"].metadata.get("file_name", "未知文档")
            if "\\" in file_name or "/" in file_name:
                file_name = Path(file_name).name
            context_parts.append(f"来源：{file_name},id:{parent_id},{parent_content}")
        
        # 4. 添加普通文档（无 parent_id 或缓存缺失）
        for idx, doc in enumerate(normal_docs):
            file_name = doc.metadata.get("file_name", f"未知文档_{idx}")
            if "\\" in file_name or "/" in file_name:
                file_name = Path(file_name).name
            context_parts.append(f"来源：{file_name}】{doc.page_content}")
        
        full_context = "\n\n".join(context_parts)
        # 可选：截断过长上下文
        if len(full_context) > 4000:
            log.warning(f"上下文过长 ({len(full_context)} 字符)，截断至 4000")
            full_context = full_context[:4000]
        print(full_context)
        return full_context

    async def stream_generate(self, question: str, docs: List[Document], session_id: Optional[str] = None) -> AsyncIterator[str]:
        """
        流式生成答案，逐步产出 token。
        注意：不存储对话历史（因为无法获得完整答案），如需存储请在外部收集完整答案后调用存储逻辑。
        """
        start_time = time.time()
        context_str = self.format_context(docs)
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
        return answer

if __name__ =='__main__':
    start = time.time()
    import asyncio
    init_thread_pools()
    vector_store = ChromaVector()
    chat_history = init_chat_history(session_id=SESSION_ID)
    llm = ChatTongyi(model=LLM_MODEL, temperature=LLM_TEMPERATURE, seed=LLM_SEED,streaming=True)

    retrieval_service = RetrievalService(vector_store, llm, chat_history)
    question = "P99响应延迟的企业要求是多少"
    docs,retriever_type =asyncio.run( retrieval_service.retrieve(question,use_context=False))
    answer_generator = AnswerGenerator(llm)

    ans =asyncio.run( answer_generator.generate(question,docs))
    end = time.time()-start
    print(f"总耗时:{end}")
    print(ans)