from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.thread_pool_manager import get_thread_pool
from config.settings import DASHSCOPE_EMBEDDING_MODEL, EMBEDDING_BACKEND, LOCAL_EMBEDDING_MODEL, VECTOR_DB_DIR
from langchain_community.embeddings import DashScopeEmbeddings,HuggingFaceEmbeddings
from infrastructure.vector_store.base_store import BaseVectorStore
from logs.log_config import data_layer_log as log
import asyncio

class ChromaVector(BaseVectorStore):
    def __init__(self, persist_dir: str = VECTOR_DB_DIR, embedding_model: str =LOCAL_EMBEDDING_MODEL) -> None:
        super().__init__()
        self.persist_dir = persist_dir
        # 嵌入模型同步/异步通用
        try:
            self._executor = get_thread_pool("vector")
        except:
            # 若还未初始化，则创建一个默认大小的线程池
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chroma")
        if EMBEDDING_BACKEND == "local":
            model_name = embedding_model or LOCAL_EMBEDDING_MODEL
            log.info(f"初始化本地嵌入模型: {model_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            model_name = embedding_model or DASHSCOPE_EMBEDDING_MODEL
            log.info(f"初始化云端嵌入模型: {model_name}")
            self.embedding_model = DashScopeEmbeddings(model=model_name)
        self.vector_store: Optional[Chroma] = None
        self._init_vector()

    def _init_vector(self):
        """初始化向量库（同步）"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model,
                collection_name="Chroma_db"
            )
            log.info(f"Chroma向量数据库加载成功,持久化目录{self.persist_dir}")
        except Exception as e:
            log.error(f"Chroma向量数据库加载失败{e}", exc_info=True)
            raise

    # ===================== 【原有同步方法 · 保留】 =====================
    def add_documents(self, docs: List[Document], batch_size: int = 30) -> None:
        """同步批量添加文档（本地用）"""
        if not docs:
            log.warning("无文档需要加载，跳过")
            return
        total = len(docs)
        for i in range(0, total, batch_size):
            batch = docs[i:i+batch_size]
            try:
                self.vector_store.add_documents(batch)
                log.info(f"Chroma 入库批次 {i//batch_size + 1}/{(total-1)//batch_size + 1}，共 {len(batch)} 个块")
            except Exception as e:
                log.error(f"批次 {i//batch_size + 1} 入库失败: {e}", exc_info=True)
                raise
        log.info(f"Chroma 入库完成，总计 {total} 个块")

    def _get_all_documents(self) -> List[Document]:
        """获取所有文档（构建BM25用）"""
        try:
            result = self.vector_store.get()
            return [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(result['documents'], result['metadatas'])
            ]
        except Exception as e:
            log.error(f"Chroma 获取所有文档失败: {e}", exc_info=True)
            return []

    def similarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Document]:
        """【修复BUG】同步相似度检索"""
        try:
            return self.vector_store.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            log.error(f"同步相似度检索失败: {e}", exc_info=True)
            return []

    def similarity_search_with_score(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Tuple[Document, float]]:
        """【修复BUG】同步带分数检索"""
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
        except Exception as e:
            log.error(f"同步带分数检索失败: {e}", exc_info=True)
            return []

    def as_retriever(self, **kwargs):
        return self.vector_store.as_retriever(**kwargs)

    def clear(self) -> None:
        """同步清空向量库"""
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
            self._init_vector()
            log.warning("Chroma 向量库已清空并重建")
        except Exception as e:
            log.error(f"向量数据库清除失败{e}", exc_info=True)
            raise

    # ===================== 【新增异步方法 · API高并发专用】 =====================
    async def aadd_documents(self, docs: List[Document], batch_size: int = 10) -> None:
        """
        ✅ 异步批量添加文档（通过线程池执行同步的 add_documents）
        用于：API 高并发、批量文档异步入库
        """
        if not docs:
            return

        loop = asyncio.get_running_loop()
        total = len(docs)
        
        # 准备数据
        ids = [doc.metadata.get("id") for doc in docs]
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        # 生成 embeddings（同步方法，放入线程池）
        embeddings = await loop.run_in_executor(
            None,
            self.embedding_model.embed_documents,
            texts
        )
        
        # 分批 upsert
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            await loop.run_in_executor(
                self._executor,
                lambda: self.vector_store._collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
            )
            log.info(f"【异步】入库批次 {i//batch_size + 1}/{(total-1)//batch_size + 1} 完成，共 {len(batch_ids)} 个块")
        
        log.info(f"【异步】Chroma upsert 完成，总计 {total} 个块")

    async def asimilarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Document]:
        """✅ 异步相似度检索（通过线程池执行同步的 similarity_search）"""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor, 
                self.vector_store.similarity_search, 
                query, 
                k=k, 
                filter=filter
            )
        except Exception as e:
            log.error(f"【异步】相似度检索失败: {e}", exc_info=True)
            return []

    async def asimilarity_search_with_score(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Tuple[Document, float]]:
        """✅ 异步带分数检索（通过线程池执行同步的 similarity_search_with_score）"""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor,
                self.vector_store.similarity_search_with_score,
                query,
                k=k,
                filter=filter
            )
        except Exception as e:
            log.error(f"【异步】带分数检索失败: {e}", exc_info=True)
            return []

    async def asimilarity_search_by_vector(self, embedding: List[float], k: int = 3, filter: Optional[dict] = None):
        """✅ 异步根据预计算向量进行相似度检索（已经正确实现，无需修改）"""
        loop = asyncio.get_running_loop()
        func = partial(self.vector_store.similarity_search_by_vector_with_relevance_scores, embedding, k=k, filter=filter)
        return await loop.run_in_executor(self._executor, func)

    async def aclear(self) -> None:
        """✅ 异步清空向量库（通过线程池执行同步的 delete_collection）"""
        loop = asyncio.get_running_loop()
        try:
            if self.vector_store:
                # delete_collection 是同步方法，放入线程池
                await loop.run_in_executor(self._executor, self.vector_store.delete_collection)
            # _init_vector 是同步方法，也放入线程池执行（或者直接调用，但为了避免阻塞，也放线程池）
            await loop.run_in_executor(self._executor, self._init_vector)
            log.info("【异步】Chroma 向量库已清空并重建")
        except Exception as e:
            log.error(f"【异步】向量数据库清除失败{e}", exc_info=True)
            raise

    async def delete_by_file(self, file_path: str):
        """异步删除指定文件的所有向量（基于 metadata 中的 file_path）"""
        loop = asyncio.get_running_loop()
        try:
            # Chroma 的 delete 支持 where 条件
            await loop.run_in_executor(
                self._executor,
                lambda: self.vector_store._collection.delete(where={"file_path": file_path})
            )
            log.info(f"已删除文件 {file_path} 的旧向量")
        except Exception as e:
            log.error(f"删除文件 {file_path} 向量失败: {e}")

    async def delete_by_ids(self, ids: List[str]):
        """批量删除指定 ID 的向量（Chroma 原生支持）"""
        if not ids:
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                lambda: self.vector_store._collection.delete(ids=ids)
            )
            log.info(f"已删除 {len(ids)} 个向量")
        except Exception as e:
            log.error(f"批量删除向量失败: {e}")
            # 不抛出，避免中断整个流程
