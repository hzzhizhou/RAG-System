# core/deployment_layer.py
import time
import hashlib
import uuid
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
import uvicorn
import os
from config.settings import API_HOST, API_PORT, CACHE_MAXSIZE, RESPONSE_TIMEOUT, DASHSCOPE_API_KEY
from logs.log_config import log
from utils.security import DataSecurity
from core.agent_layer import create_agent_with_memory
from core.chat_history_factory import init_chat_history
from langchain_community.chat_models.tongyi import ChatTongyi
from fastapi import BackgroundTasks
import asyncio
# ---------- API 请求/响应模型 ----------
class RAGRequest(BaseModel):
    question: str
    route_mode: str = "rule"
    api_key: str = DASHSCOPE_API_KEY
    session_id: Optional[str] = None
    use_context: bool = True
    use_hyde: bool = False
    use_multi: bool = False

class RAGResponse(BaseModel):
    answer: str
    retriever_type: str
    # context_chunks: List[dict]
    response_time: float
    session_id: str
    status: str = "success"

class AgentRequest(BaseModel):
    question: str
    api_key: str = DASHSCOPE_API_KEY
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    response_time: float
    session_id: str
    status: str = "success"

# ---------- 核心服务 ----------
class RAGService:
    def __init__(self, retrieval_service, answer_generator, llm: ChatTongyi):
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator
        self.llm = llm
        self.hybrid_retriever = retrieval_service.hybrid_retriever   # 获取混合检索器
        self.app = FastAPI(title="企业级RAG服务", version="1.0")
        self.cache: Dict[str, RAGResponse] = {}
        self.cache_maxsize = CACHE_MAXSIZE
        self.agent_executors: Dict[str, object] = {}
        self._register_routes()

    def _get_agent_executor(self, session_id: str):
        """获取或创建会话专用的 Agent 执行器"""
        if session_id not in self.agent_executors:
            # 创建 Agent 执行器（传入 hybrid_retriever 和 llm）
            self.agent_executors[session_id] = create_agent_with_memory(
                hybrid_retriever=self.hybrid_retriever,
                llm=self.llm
            )
            log.info(f"为会话 [{session_id}] 创建 Agent 执行器")
        return self.agent_executors[session_id]
    def _store_history(self, session_id: str, question: str, answer: str):
            """后台任务：存储对话历史到 Redis"""
            try:
                chat_history = init_chat_history(session_id)
                from langchain_core.messages import HumanMessage, AIMessage
                chat_history.add_message(HumanMessage(content=question))
                chat_history.add_message(AIMessage(content=answer))
                log.info(f"流式对话历史已存储 | 会话={session_id}")
            except Exception as e:
                log.error(f"存储对话历史失败: {e}")

    def _register_routes(self):
        # ========== 非流式 RAG 路由（保持原接口）==========
        @self.app.post("/rag/query", response_model=RAGResponse)
        async def rag_query(request: RAGRequest):
            # 鉴权
            if not DataSecurity.validate_api_key(request.api_key):
                raise HTTPException(status_code=401, detail="无效的API密钥")

            session_id = request.session_id or str(uuid.uuid4())
            cache_key = None
            if request.session_id is None:
                cache_key = self._get_cache_key(request.question, request.route_mode)
                cached = self._get_cached_result(cache_key)
                if cached:
                    return cached

            start_time = time.time()
            try:
                docs, retriever_type = await self.retrieval_service.retrieve(
                    question=request.question,
                    route_mode=request.route_mode,
                    use_context=request.use_context,
                    use_hyde=request.use_hyde,
                    use_multi=request.use_multi,
                    session_id=session_id
                )
                answer = await self.answer_generator.generate(
                    question=request.question,
                    docs=docs,
                    session_id=session_id
                )
                # chunks = [
                #     {
                #         "chunk": doc.page_content[:200] + "...",
                #         "file_name": doc.metadata.get("file_name", "unknown"),
                #         "chunk_id": doc.metadata.get("chunk_id", "")
                #     } for doc in docs
                # ]
                response = RAGResponse(
                    answer=answer,
                    retriever_type=retriever_type,
                    # context_chunks=chunks,
                    response_time=round(time.time() - start_time, 2),
                    session_id=session_id
                )
                if cache_key:
                    self._cache_result(cache_key, response)
                log.info(f"RAG 查询完成 | 会话={session_id} | 耗时={response.response_time}s")
                return response
            except Exception as e:
                log.error(f"RAG 处理失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="服务内部错误")

        # ========== 流式 RAG 路由（新增）==========
        @self.app.post("/rag/stream")
        async def rag_stream(request: RAGRequest, background_tasks: BackgroundTasks):
            # 鉴权
            if not DataSecurity.validate_api_key(request.api_key):
                raise HTTPException(status_code=401, detail="无效的API密钥")

            session_id = request.session_id or str(uuid.uuid4())


            try:
                docs, retriever_type = await self.retrieval_service.retrieve(
                    question=request.question,
                    route_mode=request.route_mode,
                    use_context=request.use_context,
                    use_hyde=request.use_hyde,
                    use_multi=request.use_multi,
                    session_id=session_id
                )
                headers = {"X-Retriever-Type": retriever_type}

                # 定义一个异步生成器，同时收集完整答案
                async def generate_and_collect():
                    full_answer = []
                    async for chunk in self.answer_generator.stream_generate(request.question, docs, session_id):
                        full_answer.append(chunk)
                        yield chunk
                    # 在生成结束后，使用 background_tasks 存储历史
                    # 注意：background_tasks 需要传入 callable，不能直接 await
                    background_tasks.add_task(
                        self._store_history, 
                        session_id, 
                        request.question, 
                        "".join(full_answer)
                    )

                ans = StreamingResponse(
                    generate_and_collect(),
                    media_type="text/plain",
                    headers=headers
                )
                return ans
            except Exception as e:
                log.error(f"流式 RAG 处理失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="服务内部错误")


        # ========== Agent 路由 ==========
        @self.app.post("/agent/query", response_model=AgentResponse)
        async def agent_query(request: AgentRequest, background_tasks: BackgroundTasks):
            # 鉴权
            if not DataSecurity.validate_api_key(request.api_key):
                raise HTTPException(status_code=401, detail="无效的API密钥")

            session_id = request.session_id or str(uuid.uuid4())
            start_time = time.time()
            try:
                # 获取 Agent 执行器
                agent_executor = self._get_agent_executor(session_id)
                # 获取对话历史（用于填充 prompt）
                chat_history = init_chat_history(session_id)
                history_messages = chat_history.messages()
                # 格式化历史为字符串（最近6条）
                history_str = ""
                for msg in history_messages[-6:]:
                    role = "用户" if isinstance(msg, HumanMessage) else "助手"
                    history_str += f"{role}：{msg.content}\n"

                # 同步执行 Agent（使用线程池避免阻塞事件循环）
                def run_agent():
                    return agent_executor.invoke({
                        "input": request.question,
                        "chat_history": history_str
                    })
                result = await asyncio.to_thread(run_agent)
                answer = result["output"]
                elapsed = round(time.time() - start_time, 2)

                # 存储用户问题和助手回答到历史（后台任务）
                background_tasks.add_task(chat_history.add_message, HumanMessage(content=request.question))
                background_tasks.add_task(chat_history.add_message, AIMessage(content=answer))

                log.info(f"Agent 查询完成 | 会话={session_id} | 耗时={elapsed}s")
                return AgentResponse(
                    answer=answer,
                    response_time=elapsed,
                    session_id=session_id,
                    status="success"
                )
            except Exception as e:
                log.error(f"Agent 处理失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Agent 服务内部错误")

    # ---------- 缓存辅助方法 ----------
    def _get_cache_key(self, question: str, route_mode: str) -> str:
        q_hash = hashlib.md5(question.encode()).hexdigest()
        return f"{q_hash}_{route_mode}"

    def _cache_result(self, key: str, result: RAGResponse):
        if len(self.cache) >= self.cache_maxsize:
            oldest = next(iter(self.cache.keys()))
            del self.cache[oldest]
        self.cache[key] = result

    def _get_cached_result(self, key: str):
        return self.cache.get(key)

    def run(self):
        log.info(f"启动RAG API服务：http://{API_HOST}:{API_PORT}")
        uvicorn.run(
            self.app,
            host=API_HOST,
            port=API_PORT,
            timeout_keep_alive=RESPONSE_TIMEOUT
        )