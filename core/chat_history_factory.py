from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from typing import List
from infrastructure.redis.redis_history import RedisChatHistory
from logs.log_config import chat_history_log as log
import time
def init_chat_history(session_id:str):
    start_time = time.time()
    try:
        redis_chat_history = RedisChatHistory(session_id)
        end_time = time.time()
        redis_time = end_time - start_time
        log.info(f"Redis 连接成功，耗时：{redis_time:.2f}秒")
        return redis_chat_history
    except Exception as e:
        log.error(f"Redis 连接失败，启用内存存储{e}")
        from langchain_core.chat_history import InMemoryChatMessageHistory
        return InMemoryChatMessageHistory()

if __name__=='__main__':
    from langchain_core.messages import HumanMessage, AIMessage

    # 准备消息列表
    raw_messages = [
        ("human", "doaghih")
    ]

    redis_chat_history = init_chat_history("user_001")
    # 逐个添加并转换
    # redis_chat_history.clear()
    print(redis_chat_history.messages())
    # redis_chat_history.add_message(HumanMessage("何乐乐胡覅"))
