"""
实现文本数据加载，分块和向量数据库存储，向量数据库初始化
"""
import asyncio
from pathlib import Path
import sys
import time

from langchain_core.documents import Document
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CHUNKING_STRATEGY
from logs.log_config import data_layer_log as log
BASE_DIR = Path(__file__).parent.parent
from document_splitter.splitter_factory import get_chunker
from data_loader.data_loader import DocumentLoader
class DataLayer:
    @staticmethod
    async def data_loader(file_path, total_timeout: int = 300):
        try:
            # 总超时控制
            await asyncio.wait_for(
                DataLayer._inner_data_loader(file_path),
                timeout=total_timeout
            )
        except asyncio.TimeoutError:
            log.error(f"总加载流程超时（{total_timeout}秒）")
            raise
        except Exception as e:
            log.error(f"入库失败{e}", exc_info=True)
            raise

    @staticmethod
    async def _inner_data_loader(file_path):
        # --------------------第一步：文档加载---------------
        start_time = time.time()
        documentloader = DocumentLoader()
        docs = await documentloader.dir_loader(file_path)
        end_time = time.time()
        loader_time = end_time - start_time
        log.info(f"文档加载成功，耗时:{loader_time:.2f}秒（共{len(docs)}个原始文档）")
        
        #--------------------第二步:文档分块----------------------
        start_time = time.time()
        splitter = get_chunker(CHUNKING_STRATEGY)
        result = splitter.split_documents(docs)
        #组合策略的result返回的是一个tuple，需要单独处理
        if isinstance(result, tuple) and len(result) >= 2:
            children_list = result[0]   # 子块列表
            parent_map = result[1]      # 父块映射
        else:
            children_list = result
            parent_map = {}

    # 如果是组合分块，保存父块映射
        if CHUNKING_STRATEGY == "combined_splitter" and parent_map:
            import json
            parent_map_path = BASE_DIR / "parent_cache.json"
            with open(parent_map_path, "w", encoding="utf-8") as f:
                json.dump(parent_map, f, ensure_ascii=False, indent=2)
            log.info(f"父块映射已保存到 {parent_map_path}")

        end_time = time.time()
        split_time = end_time - start_time
        log.info(f"分块完成,耗时:{split_time:.2f}秒（共{len(children_list)}个块）")
        
        #----------------第三步:向量化，入库-------------
        from infrastructure.vector_store.async_chroma_vector import ChromaVector
        start_time = time.time()
        vector = ChromaVector()
        vector.clear()
        await vector.aadd_documents(children_list)
        # vector.add_documents(chunks)
        end_time = time.time()
        embedding_time = end_time - start_time
        log.info(f"向量索引构建完成，耗时:{embedding_time:.2f}秒")
        log.info("✅ 文档入库成功")

if __name__=='__main__':
    start_time = time.time()
    # file_path = BASE_DIR/"data"/"ragas.md"
    # asyncio.run(DataLayer.data_loader(file_path))

    dir_path = BASE_DIR / "data"
    asyncio.run(DataLayer.data_loader(dir_path))
    end_time = time.time()-start_time
    print(f'耗时:{end_time}')