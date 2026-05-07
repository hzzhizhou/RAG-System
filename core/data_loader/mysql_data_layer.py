from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import asyncio
from config.settings import BASE_DIR
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from infrastructure.sql.mysql_state_manager import MySQLStateManager
from core.data_loader.file_watcher import FileWatcher

async def main():
    vector_store = ChromaVector()
    state_mgr = MySQLStateManager()
    await state_mgr.initialize()
    
    watcher = FileWatcher(
        watch_dir=BASE_DIR / "data",
        vector_store=vector_store,
        state_mgr=state_mgr
    )
    await watcher.initialize()
    await watcher.start()
    
    # 保持服务运行，直到收到 Ctrl+C
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await watcher.stop()

if __name__ == "__main__":
    asyncio.run(main())