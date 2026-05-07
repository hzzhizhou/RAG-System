# 在文件头部添加导入
from infrastructure.sql.mysql_state_manager import MySQLStateManager
import hashlib
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

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import BASE_DIR, CHUNKING_STRATEGY
from core.document_splitter.splitter_factory import get_chunker
from logs.log_config import data_layer_log as log


MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".md"]
THREAD_POOL = ThreadPoolExecutor(max_workers=8)
class StreamDocumentLoader:
    def __init__(self):
        self.splitter = get_chunker(CHUNKING_STRATEGY)
        self.state_mgr: MySQLStateManager = None   # 改为 MySQL
        self.accumulated_parent_map = {}

    async def initialize(self):
        """初始化状态管理器（必须在 stream_dir_loader 前调用）"""
        self.state_mgr = MySQLStateManager()
        await self.state_mgr.initialize()
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
        raw_path = chunk.metadata.get("file_path", chunk.metadata.get("source", ""))
        if not raw_path:
            return "unknown_0"
        abs_path = Path(raw_path).resolve().as_posix()   # 例如 D:/data/test.txt
        chunk_index = chunk.metadata.get("chunk_index", 0)
        return f"{abs_path}_{chunk_index}"

    async def stream_load_one_file(self, file_path: Path, vector_store):
        # 检查是否需要更新
        if not await self.state_mgr.need_update(file_path):
            log.info(f"文件无变化，跳过：{file_path.name}")
            return

        try:
            # 1. 删除旧向量（基于 MySQL 中记录的旧 chunk_id）
            old_chunk_ids = await self.state_mgr.get_old_chunk_ids(file_path)
            if old_chunk_ids:
                await vector_store.delete_by_ids(old_chunk_ids)

            # 2. 加载、分块（原有逻辑，不变）
            docs = await self.single_file_loader(file_path)
            if not docs:
                return
            result = self.splitter.split_documents(docs)
            if isinstance(result, tuple):
                chunks = result[0]
                parent_map = result[1]
                if parent_map:
                    self.accumulated_parent_map.update(parent_map)
            else:
                chunks = result
            # 添加 id 到 metadata
            for chunk in chunks:
                chunk.metadata["id"] = self._get_chunk_id(chunk)

            # 入库
            await vector_store.aadd_documents(chunks)

            # 更新 MySQL 状态时使用同样的 chunk_ids
            chunk_ids = [chunk.metadata["id"] for chunk in chunks]
            await self.state_mgr.update_state(file_path, chunk_ids)
            # 3. 入库向量库
            await vector_store.aadd_documents(chunks)

            # 4. 更新 MySQL 状态
            chunk_ids = [self._get_chunk_id(chunk) for chunk in chunks]
            await self.state_mgr.update_state(file_path, chunk_ids)

            log.info(f"✅ 增量更新完成: {file_path.name}，共 {len(chunks)} 个块")
        except Exception as e:
            log.error(f"处理失败 {file_path.name}: {e}", exc_info=True)

    async def stream_dir_loader(self, dir_path: Path, vector_store):
        # 确保状态管理器已初始化
        if self.state_mgr is None:
            await self.initialize()

        # 获取当前文件列表
        current_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix in ALLOWED_EXTENSIONS]
        current_paths = {str(f) for f in current_files}

        # 获取 MySQL 中记录的文件路径
        recorded_paths = await self.state_mgr.get_all_file_paths()

        # 处理已删除的文件
        deleted_paths = recorded_paths - current_paths
        for del_path_str in deleted_paths:
            del_path = Path(del_path_str)
            try:
                old_ids = await self.state_mgr.get_old_chunk_ids(del_path)
                if old_ids:
                    await vector_store.delete_by_ids(old_ids)
                await self.state_mgr.remove_document(del_path)
                log.info(f"🗑️ 已清理被删除文件: {del_path.name}")
            except Exception as e:
                log.error(f"清理失败 {del_path.name}: {e}")

        # 处理新增或修改的文件
        tasks = [self.stream_load_one_file(f, vector_store) for f in current_files]
        await asyncio.gather(*tasks)

        # 保存父块映射（若使用组合分块）
        if self.accumulated_parent_map:
            import json
            parent_map_path = BASE_DIR / "parent_cache.json"
            with open(parent_map_path, "w", encoding="utf-8") as f:
                json.dump(self.accumulated_parent_map, f, ensure_ascii=False, indent=2)
            log.info(f"父块映射已保存到 {parent_map_path}")

        log.info("MySQL 增量更新全流程完成")