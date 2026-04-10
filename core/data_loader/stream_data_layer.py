import asyncio
from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).parent.parent.parent))
from logs.log_config import data_layer_log as log

BASE_DIR = Path(__file__).parent.parent.parent

class DataLayer:
    @staticmethod
    async def data_loader(file_path):
        try:
            from stream_data_loader import StreamDocumentLoader
            from document_splitter import DocumentSplitter
            from infrastructure.vector_store.async_chroma_vector import ChromaVector

            # 初始化
            vector = ChromaVector()
            vector.aclear()  
            splitter = DocumentSplitter()
            loader = StreamDocumentLoader()

            start_time = time.time()

            # ===================== 执行流式处理 =====================
            await loader.stream_dir_loader(file_path, vector, splitter)

            # 耗时统计
            total_time = time.time() - start_time
            log.info(f"✅ 流式全流程完成，总耗时：{total_time:.2f}秒")

        except Exception as e:
            log.error(f"入库失败: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    start_time = time.time()
    dir_path = BASE_DIR / "data"
    asyncio.run(DataLayer.data_loader(dir_path))
    end_time = time.time()-start_time
    print(f'耗时:{end_time}')