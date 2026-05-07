from pathlib import Path
from datetime import datetime
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredExcelLoader, UnstructuredMarkdownLoader,
    PyPDFLoader, TextLoader, Docx2txtLoader
)
import hashlib
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.state_manager_sql import SQLiteStateManager
from config.settings import BASE_DIR, CHUNKING_STRATEGY
from core.document_splitter.splitter_factory import get_chunker
from logs.log_config import data_layer_log as log


MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".md"]
THREAD_POOL = ThreadPoolExecutor(max_workers=8)

class StreamDocumentLoader:
    def __init__(self, db_path: Path = BASE_DIR / "doc_state.db"):
        # 分块器只创建一次，所有文件共用
        self.splitter = get_chunker(CHUNKING_STRATEGY)
        # 父块映射累积（如果是组合分块）
        self.accumulated_parent_map = {}
        self.state_mgr = SQLiteStateManager(db_path)

    # 静态方法生成 chunk_id（与向量库中的 ID 保持一致）
    @staticmethod
    def _get_chunk_id(chunk: Document) -> str:
        file_path = chunk.metadata.get("file_path", chunk.metadata.get("source", "unknown"))
        chunk_index = chunk.metadata.get("chunk_index", 0)
        unique_str = f"{file_path}_{chunk_index}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    # 同步文件加载（供线程池调用）
    def _load_sync(self, file_path: Path):
        try:
            if file_path.suffix == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_path.suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_path.suffix == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_path.suffix == ".xlsx":
                loader = UnstructuredExcelLoader(file_path)
            elif file_path.suffix == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError("不支持的格式")

            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.strip("."),
                    "file_size": str(file_path.stat().st_size),
                    "load_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                # 转换 Path 对象
                if 'source' in doc.metadata and isinstance(doc.metadata['source'], Path):
                    doc.metadata['source'] = str(doc.metadata['source'])
            from utils.text_clean import clean_documents
            return clean_documents(docs)
        except Exception as e:
            log.error(f"加载失败 {file_path.name}: {e}")
            return []

    # 异步单文件加载
    async def single_file_loader(self, file_path: Path) -> list[Document]:
        if not file_path.exists() or file_path.suffix not in ALLOWED_EXTENSIONS:
            return []
        if file_path.stat().st_size > MAX_FILE_SIZE:
            log.warning(f"文件过大跳过: {file_path.name}")
            return []
        log.info(f"⏳ 流式加载: {file_path.name}")
        docs = await asyncio.get_event_loop().run_in_executor(
            THREAD_POOL, self._load_sync, file_path
        )
        return docs
    @staticmethod
    def _get_chunk_id(chunk: Document) -> str:
        file_path = chunk.metadata.get("file_path", chunk.metadata.get("source", "unknown"))
        chunk_index = chunk.metadata.get("chunk_index", 0)
        unique_str = f"{file_path}_{chunk_index}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    async def stream_load_one_file(self, file_path: Path, vector_store):
        if not self.state_mgr.need_update(file_path):
            log.info(f"文件无变化，跳过：{file_path.name}")
            return
        
        try:
            # 1. 获取旧 chunk_id 并删除向量
            old_chunk_ids = self.state_mgr.get_old_chunk_ids(file_path)
            if old_chunk_ids:
                await vector_store.delete_by_ids(old_chunk_ids)
                log.info(f"已删除 {len(old_chunk_ids)} 个旧块: {file_path.name}")
            
            # 2. 加载文档
            docs = await self.single_file_loader(file_path)
            if not docs:
                # 如果加载失败（如文件损坏），不更新状态，避免丢失旧的记录
                log.warning(f"加载文档失败，跳过更新: {file_path.name}")
                return
            
            # 3. 分块
            result = self.splitter.split_documents(docs)
            if isinstance(result, tuple) and len(result) >= 2:
                chunks = result[0]
                parent_map = result[1]
                if parent_map:
                    self.accumulated_parent_map.update(parent_map)
            else:
                chunks = result
            
            # 4. 入库（向量库）
            await vector_store.aadd_documents(chunks)
            
            # 5. 记录新 chunk_id 到状态库
            chunk_ids = [self._get_chunk_id(chunk) for chunk in chunks]
            self.state_mgr.update_state(file_path, chunk_ids)
            log.info(f"✅ 增量更新完成: {file_path.name}，共 {len(chunks)} 个块")
        except Exception as e:
            log.error(f"处理失败 {file_path.name}: {e}", exc_info=True)
    
    async def stream_dir_loader(self, dir_path: Path, vector_store):
        # 1. 获取当前目录下的有效文件
        current_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix in ALLOWED_EXTENSIONS]
        current_paths = {str(f) for f in current_files}
        
        # 2. 获取状态库中记录的所有文件路径
        recorded_paths = self.state_mgr.get_all_file_paths()
        
        # 3. 处理已删除的文件
        deleted_paths = recorded_paths - current_paths
        for del_path_str in deleted_paths:
            del_path = Path(del_path_str)
            try:
                # 删除该文件的所有 chunk
                old_ids = self.state_mgr.get_old_chunk_ids(del_path)
                if old_ids:
                    await vector_store.delete_by_ids(old_ids)
                self.state_mgr.remove_document(del_path)
                log.info(f"🗑️ 已清理被删除文件: {del_path.name}")
            except Exception as e:
                log.error(f"清理失败 {del_path.name}: {e}")
        
        # 4. 并发处理当前文件（新增/修改）
        tasks = [self.stream_load_one_file(f, vector_store) for f in current_files]
        await asyncio.gather(*tasks)
        
        # 5. 保存父块映射（如果是组合分块）
        if self.accumulated_parent_map:
            parent_map_path = BASE_DIR / "parent_cache.json"
            import json
            with open(parent_map_path, "w", encoding="utf-8") as f:
                json.dump(self.accumulated_parent_map, f, ensure_ascii=False, indent=2)
            log.info(f"父块映射已保存到 {parent_map_path} (共 {len(self.accumulated_parent_map)} 个父块)")
        
        log.info("所有文件流式处理+入库完成")