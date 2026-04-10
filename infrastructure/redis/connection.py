import redis
from config.settings import REDIS_CONFIG
from logs.log_config import chat_history_log as log
_redis_pool = None

def get_redis_connection():
    """
    获取Redis连接，如果连接池尚未初始化则进行初始化，并逐行解释内部实现。
    """
    global _redis_pool  # 声明使用全局变量_redis_pool，确保可以在函数内修改它
    if _redis_pool is None:  # 判断连接池是否尚未初始化
        try:
            # 创建Redis连接池对象，参数由配置文件REDIS_CONFIG提供
            _redis_pool = redis.ConnectionPool(
                host=REDIS_CONFIG["host"],                   # Redis服务器地址
                port=REDIS_CONFIG["port"],                   # Redis服务器端口
                password=REDIS_CONFIG["password"],           # Redis密码（如有）
                db=REDIS_CONFIG["db"],                       # 使用的数据库序号
                decode_responses=True,                       # 将Redis返回的字节流自动解码为str
                socket_timeout=REDIS_CONFIG["socket_timeout"],# 连接超时时间(s)，防止长时间等待
                retry_on_timeout=True,                       # 连接超时后是否自动重试
                max_connections=REDIS_CONFIG["max_connections"] # 最大连接数
            )
            # 创建一个测试用Redis客户端，连接上面创建的连接池
            test_client = redis.Redis(connection_pool=_redis_pool)
            # 主动发送PING命令，测试Redis连接是否有效
            test_client.ping()
            # 记录日志：连接池初始化成功
            log.info("Redis 连接池初始化成功")
        except Exception as e:
            # 如果初始化或连接测试发生异常，则记录异常日志
            log.error(f"Redis 连接池初始化失败: {e}")
            # 向上抛出该异常
            raise
    # 无论是新初始化还是复用全局连接池，最后返回一个Redis客户端
    return redis.Redis(connection_pool=_redis_pool)
