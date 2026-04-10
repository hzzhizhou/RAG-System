from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredExcelLoader, UnstructuredMarkdownLoader,
    PyPDFLoader, TextLoader, Docx2txtLoader
)
import sys
import asyncio
sys.path.append(str(Path(__file__).parent.parent.parent))
from logs.log_config import data_layer_log as log

BASE_DIR = Path(__file__).parent.parent
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".md"]

# 线程池：解决同步加载阻塞异步的问题
from concurrent.futures import ThreadPoolExecutor
THREAD_POOL = ThreadPoolExecutor(max_workers=8)

class StreamDocumentLoader:
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
            # 元数据
            for doc in docs:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.strip("."),
                    "file_size": str(file_path.stat().st_size),
                    "load_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            # 清洗
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
        # 使用线程池将同步的 _load_sync 方法转为异步执行，防止阻塞事件循环。
        # asyncio.get_event_loop().run_in_executor 会在后台线程中调用 self._load_sync(file_path)，
        # 并异步等待结果，相当于将同步的I/O操作和文本处理异步化，适用于阻塞型操作（如大文件加载）。
        docs = await asyncio.get_event_loop().run_in_executor(
            THREAD_POOL, self._load_sync, file_path
        )
        return docs

    # ===================== 【流式处理核心】 =====================
    async def stream_load_one_file(self, file_path: Path, vector_store, splitter):
        """
        单个文件完整流式流水线：
        加载 → 清洗 → 分块 → 异步Embedding入库
        """
        try:
            # 1. 加载+清洗
            docs = await self.single_file_loader(file_path)
            if not docs:
                return

            # 2. 分块
            chunks = splitter.split_documents(docs)
            log.info(f"📦 {file_path.name} 分块完成：{len(chunks)}块")

            # 3. 立刻异步入库（不等其他文件！）
            await vector_store.aadd_documents(chunks)
            log.info(f"✅【流式完成】{file_path.name} 全流程结束")

        except Exception as e:
            log.error(f"❌ {file_path.name} 流式处理失败: {e}")

    # 批量流式加载（并发执行所有文件）
    async def stream_dir_loader(self, dir_path: Path, vector_store, splitter):
        files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix in ALLOWED_EXTENSIONS]
        log.info(f"启动流式处理，共 {len(files)} 个文件")

        # 并发执行：每个文件独立流水线
        tasks = [
            self.stream_load_one_file(f, vector_store, splitter)
            for f in files
        ]
        await asyncio.gather(*tasks)
        log.info("所有文件流式处理+入库全部完成")