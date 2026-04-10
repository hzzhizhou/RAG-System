from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
sys.path.append(str(Path(__file__).parent.parent.parent))
from logs.log_config import data_layer_log as log
BASE_DIR = Path(__file__).parent.parent
MAX_FILE_SIZE = 10 * 1024 * 1024  # 最大文件大小（10MB，规避内存溢出）
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx",".md"]  # 允许的文件格式

# 线程池（用于将同步IO转为异步）
THREAD_POOL_EXECUTOR = ThreadPoolExecutor(max_workers=8)  # 根据CPU核心数调整

# --------------------第一步：文档加载---------------
class DocumentLoader:
    # 封装同步加载逻辑为独立函数（供线程池调用）
    def _sync_load_file(self, file_path: Path):
        try:
            log.info(f"⏳ 开始加载：{file_path.name}")
            if file_path.suffix == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            elif file_path.suffix == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_path.suffix == ".docx":  
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif file_path.suffix == ".xlsx":
                loader = UnstructuredExcelLoader(file_path)
                docs = loader.load()
            elif file_path.suffix == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
            else:
                raise ValueError("不支持的文件加载")
            
            # 添加元数据
            for doc in docs:
                doc.metadata.update({
                    "file_name": str(file_path.name),
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lstrip("."),
                    "file_size": str(file_path.stat().st_size),
                    "load_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            # 强制将 source 字段（由 UnstructuredMarkdownLoader 产生）转换为字符串
            if 'source' in doc.metadata and isinstance(doc.metadata['source'], Path):
                doc.metadata['source'] = str(doc.metadata['source'])
            # 数据清洗
            from utils.text_clean import clean_documents
            cleaned_docs = clean_documents(docs)
            log.info(f"✅ 加载完成：{file_path.name}")
            return cleaned_docs
        except Exception as e:
            log.error(f"❌ 文件加载失败 {file_path.name}，错误:{e}")
            raise

    #-----------单文件加载实现（真正异步）---------
    async def single_file_loader(self, file_path: Path, timeout: int = 10):
        if not file_path.exists():
            log.error(f"文件不存在：{file_path}")
            raise FileNotFoundError(f"文件不存在:{file_path}")
        
        if file_path.suffix not in ALLOWED_EXTENSIONS:
            log.error(f"不支持该文件类型{file_path.suffix}，请传入{ALLOWED_EXTENSIONS}类型的文件")
            raise ValueError(f"不支持的文件格式{file_path.suffix}")
        
        if file_path.stat().st_size > MAX_FILE_SIZE:
            log.error(f"文件大小超出{MAX_FILE_SIZE/1024/1024}MB限制")
            raise ValueError(f"{file_path}文件太大,超出{MAX_FILE_SIZE/1024/1024}MB的限制")
        
        try:
            # 将同步加载逻辑提交到线程池，实现真正异步
            cleaned_docs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(THREAD_POOL_EXECUTOR, self._sync_load_file, file_path),
                timeout=timeout  # 超时控制
            )
            return cleaned_docs
        except asyncio.TimeoutError:
            log.error(f"⏰ 文件加载超时（{timeout}秒）：{file_path.name}")
            raise
        except Exception as e:
            log.error(f"文件加载失败，错误:{e}")
            raise

    #-----------一次加载多个文件（真正并发）----------------
    async def dir_loader(self, dir_path: Path):
        all_docs = []
        if not dir_path.exists():
            log.error(f"批量加载失败,目录文件{dir_path}不存在")
            raise FileNotFoundError(f"批量加载失败,目录文件{dir_path}不存在")
        
        # 缓存文件列表（避免重复遍历+结果匹配错误）
        files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix in ALLOWED_EXTENSIONS]
        if not files:
            log.warning(f"目录{dir_path}下无符合要求的文件")
            return all_docs
        
        log.info(f"🚀 并发启动 {len(files)} 个文件加载（异步+线程池模式）")
        # 创建异步任务
        tasks = [self.single_file_loader(file) for file in files]
        # 并发执行（return_exceptions=True 不中断其他任务）
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果（用缓存的files列表匹配，避免路径错误）
        for file, result in zip(files, results):
            if isinstance(result, Exception):
                log.warning(f"跳过异常文件：{file.name}，错误：{str(result)}")
            else:
                all_docs.extend(result)
        
        log.info(f"📊 加载完成，成功加载 {len(all_docs)} 个文档（共处理 {len(files)} 个文件）")
        return all_docs

