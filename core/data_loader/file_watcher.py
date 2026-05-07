import asyncio
from pathlib import Path
from watchfiles import awatch, Change
from logs.log_config import data_layer_log as log
from core.data_loader.mysql_data_loader import StreamDocumentLoader
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from infrastructure.sql.mysql_state_manager import MySQLStateManager

class FileWatcher:
    def __init__(self, watch_dir: Path, vector_store: ChromaVector, state_mgr: MySQLStateManager):
        self.watch_dir = watch_dir.resolve()
        self.vector_store = vector_store
        self.state_mgr = state_mgr
        self.loader = StreamDocumentLoader()
        self._task = None
        self._running = False

    async def initialize(self):
        """初始化加载器（内部会初始化其状态管理器）"""
        await self.loader.initialize()

    async def start(self):
        log.info(f"🚀 开始监听目录: {self.watch_dir}")
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        # 可选：启动时进行一次全量追赶扫描
        asyncio.create_task(self.initial_sync())

    async def _watch_loop(self):
        log.info("👀 监控循环已开始，等待文件变化...") 
        try:
            async for changes in awatch(str(self.watch_dir), recursive=True):
                log.info(f"📡 收到原始事件: {changes}")   # 添加这行
                for change_type, file_path in changes:
                    asyncio.create_task(self._handle_change(change_type, Path(file_path)))
        except Exception as e:
            log.error(f"监控循环异常: {e}", exc_info=True)

    async def _handle_change(self, change_type: Change, file_path: Path):
        # 防抖：等待编辑器/文件系统稳定
        await asyncio.sleep(2)
        try:
            if change_type == Change.added:
                await self.loader.stream_load_one_file(file_path, self.vector_store)
                log.info(f"🆕 已处理新增文件: {file_path.name}")
            elif change_type == Change.modified:
                await self.loader.stream_load_one_file(file_path, self.vector_store)
                log.info(f"✏️ 已处理修改文件: {file_path.name}")
            elif change_type == Change.deleted:
                # 1. 获取该文件的所有旧 chunk_id
                log.info(f"准备删除文件: {file_path}, 绝对路径: {file_path.resolve()}")
                old_chunk_ids = await self.state_mgr.get_old_chunk_ids(file_path)
                log.info(f"获取到旧 chunk_ids: {old_chunk_ids}")
                # 2. 从向量库删除这些 chunk
                collection = self.vector_store.vector_store._collection
                doc = collection.get(ids=[old_chunk_ids[0]])
                if doc['ids']:
                    log.info(f"该 chunk_id 存在于向量库中")
                else:
                    log.warning("chunk_id 不在向量库中，可能之前已被删除或从未存在")
                if old_chunk_ids:
                    await self.vector_store.delete_by_ids(old_chunk_ids)
                    log.info(f"🗑️ 已从向量库删除 {len(old_chunk_ids)} 个块: {file_path.name}")
                # 3. 再从 MySQL 删除文档记录
                await self.state_mgr.remove_document(file_path)
                log.info(f"🗑️ 已从数据库移除文件记录: {file_path.name}")
        except Exception as e:
            log.error(f"❌ 处理文件 {file_path.name} 变更失败: {e}", exc_info=True)

    async def initial_sync(self):
        log.info("🐌 执行启动追赶扫描...")
        await self.loader.stream_dir_loader(self.watch_dir, self.vector_store)
        log.info("✅ 追赶扫描完成")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            await self._task
        await self.state_mgr.close()