# tools/webpage_fetcher.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import requests
from trafilatura import extract
from langchain_core.tools import tool
from infrastructure.redis.redis_case import redis_cache
from logs.log_config import log
from utils.metrics import fetch_calls_total, fetch_cache_hits, fetch_cache_misses
@tool(description="抓取指定 URL 的网页正文，返回纯文本。适用于用户提供了具体链接的情况。")
def fetch_webpage(url: str) -> str:
    fetch_calls_total.inc()
    cached = redis_cache.get_webpage(url)
    if cached:
        fetch_cache_hits.inc()
        log.info(f"url缓存命中:{url}")
        return cached
    fetch_cache_misses.inc()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = extract(resp.text, include_comments=False, include_tables=True)
        if not text:
            return "无法提取网页正文"
        result = text[:2000]  # 限制长度
        redis_cache.set_webpage(url, result)
        return result
    except requests.RequestException as e:
        return f"网页抓取失败: {str(e)}"

if __name__ == '__main__':
    docs = fetch_webpage.run("https://jtj.shaoyang.gov.cn/syjtj/shzyhejzg/202011/289dced0e96c48129aeefa9019bd061d.shtml")
    print(docs)