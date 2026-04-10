from pathlib import Path
from datetime import datetime
import sys
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredExcelLoader, UnstructuredMarkdownLoader,
    PyPDFLoader, TextLoader, Docx2txtLoader
)

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CHUNKING_STRATEGY
from core.document_splitter.splitter_factory import get_chunker
from logs.log_config import data_layer_log as log

BASE_DIR = Path(__file__).parent.parent
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".md"]
THREAD_POOL = ThreadPoolExecutor(max_workers=8)


class StreamDocumentLoader:
    def __init__(self):
        # 分块器只创建一次，所有文件共用
        self.splitter = get_chunker(CHUNKING_STRATEGY)
        # 父块映射累积（如果是组合分块）
        self.accumulated_parent_map = {}

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

    # 单个文件完整流水线（加载 → 分块 → 入库）
    async def stream_load_one_file(self, file_path: Path, vector_store):
        try:
            # 1. 加载文档
            docs = await self.single_file_loader(file_path)
            if not docs:
                return

            # 2. 分块
            result = self.splitter.split_documents(docs)
            if isinstance(result, tuple) and len(result) >= 2:
                chunks = result[0]          # 子块列表
                parent_map = result[1]      # 当前文件的父块映射
                # 累积父块映射
                if parent_map:
                    self.accumulated_parent_map.update(parent_map)
            else:
                chunks = result

            # 3. 异步入库
            await vector_store.aadd_documents(chunks)
            log.info(f"✅【流式完成】{file_path.name} 入库 {len(chunks)} 个块")

        except Exception as e:
            log.error(f"❌ {file_path.name} 流式处理失败: {e}", exc_info=True)

    # 批量流式加载（并发执行所有文件）
    async def stream_dir_loader(self, dir_path: Path, vector_store):
        files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix in ALLOWED_EXTENSIONS]
        if not files:
            log.warning("目录下无符合要求的文件")
            return

        log.info(f"启动流式处理，共 {len(files)} 个文件")
        tasks = [self.stream_load_one_file(f, vector_store) for f in files]
        await asyncio.gather(*tasks)

        # 所有文件处理完后，保存累积的父块映射（如果是组合分块）
        if self.accumulated_parent_map:
            parent_map_path = BASE_DIR / "parent_cache.json"
            with open(parent_map_path, "w", encoding="utf-8") as f:
                json.dump(self.accumulated_parent_map, f, ensure_ascii=False, indent=2)
            log.info(f"父块映射已保存到 {parent_map_path} (共 {len(self.accumulated_parent_map)} 个父块)")

        log.info("所有文件流式处理+入库全部完成")