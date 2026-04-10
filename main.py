from pathlib import Path
import sys
import atexit
from utils.thread_pool_manager import init_thread_pools
from utils.thread_pool_manager import shutdown_thread_pools
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from langchain_community.chat_models.tongyi import ChatTongyi
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from core.retriever_layer import RetrievalService
from core.generation_layer import AnswerGenerator
from core.chat_history_factory import init_chat_history
from core.deployment_layer import RAGService
from config.settings import SESSION_ID, LLM_MODEL, LLM_TEMPERATURE, LLM_SEED
def main():
    # 初始化各层
    init_thread_pools()
    atexit.register(shutdown_thread_pools)
    vector_store = ChromaVector()
    chat_history = init_chat_history(session_id=SESSION_ID)
    llm = ChatTongyi(model=LLM_MODEL, temperature=LLM_TEMPERATURE, seed=LLM_SEED,streaming=True)
    retrieval_service = RetrievalService(vector_store, llm, chat_history)
    answer_generator = AnswerGenerator(llm)
    service = RAGService(retrieval_service, answer_generator,llm)
    # 启动服务
    service.run()

if __name__ == "__main__":
    main()