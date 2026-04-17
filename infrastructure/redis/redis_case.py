from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import redis
import json
import hashlib
from config.settings import REDIS_CONFIG
from logs.log_config import log  # 如果 Web Agent 也有日志模块，可复用；否则用 print
from infrastructure.redis.connection import get_redis_connection


class RedisCache:
    def __init__(self):
        self.client = get_redis_connection()

    def _key(self, prefix: str, identifier: str) -> str:
        """生成带前缀的缓存键"""
        return f"{REDIS_CONFIG['key_prefix_web']}{prefix}:{hashlib.md5(identifier.encode()).hexdigest()}"

    def get_search(self, query: str):
        if not self.client:
            return None
        key = self._key("search", query)
        data = self.client.get(key)
        return data if data else None

    def set_search(self, query: str, result: str, ttl: int = None):
        if not self.client:
            return None
        if ttl is None:
            ttl = REDIS_CONFIG.get("cache_ttl_search", 3600)
        key = self._key("search", query)
        self.client.setex(key, ttl, result)

    def get_webpage(self, url: str):
        if not self.client:
            return None
        key = self._key("webpage", url)
        data = self.client.get(key)
        return data if data else None

    def set_webpage(self, url: str, content: str, ttl: int = None):
        if not self.client:
            return None
        if ttl is None:
            ttl = REDIS_CONFIG.get("cache_ttl_webpage", 86400)
        key = self._key("webpage", url)
        self.client.setex(key, ttl, content)

    def clear_pattern(self, pattern: str):
        """批量删除匹配前缀的键（慎用）"""
        if not self.client:
            return None
        full_pattern = f"{REDIS_CONFIG['key_prefix_web']}{pattern}*"
        keys = self.client.keys(full_pattern)
        if keys:
            self.client.delete(*keys)
            log.info(f"Deleted {len(keys)} keys with pattern {full_pattern}")

redis_cache = RedisCache()

if __name__=='__main__':
    url  ="https://jtj.shaoyang.gov.cn/syjtj/shzyhejzg/202011/289dced0e96c48129aeefa9019bd061d.shtml"
    query = "今天时政新闻"
    data = redis_cache.get_search(query)
    print(data)