from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from infrastructure.redis.redis_case import redis_cache
from logs.log_config import log
from utils.metrics import search_calls_total, search_cache_hits, search_cache_misses,search_errors_total
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SEARCH_URL = "https://google.serper.dev/search"
if not SERPER_API_KEY:
    log.error("SERPER_API_KEY 未配置，搜索功能不可用")
# 重试策略：最多3次，指数退避（1s, 2s, 4s），仅对网络相关异常重试
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def _call_search_api(query: str, headers: dict, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(SEARCH_URL, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


@tool(description="搜索互联网获取实时信息，返回相关网页摘要和链接。适用于新闻、事实查询、最新动态等。")
async def web_search(query: str) -> str:
    # 1. 尝试从 Redis 获取缓存
    search_calls_total.inc()
    log.info(f"调用 web_search，查询：{query}")
    cached = redis_cache.get_search(query)
    if cached:
        search_cache_hits.inc()
        log.info(f"search缓存命中\n{cached}")
        return cached
    search_cache_misses.inc()

    # 2. 缓存未命中，调用 API（带重试）
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": 3}
    try:
        data = await _call_search_api(query, headers, payload)
    except Exception as e:
        search_errors_total.inc()
        log.error(f"搜索失败（已重试3次）: {e}")
        return f"搜索失败: {str(e)}"

    # 3. 解析结果
    organic = data.get("organic", [])
    if not organic:
        return "未找到相关搜索结果"
    formatted = []
    for r in organic[:1]:
        title = r.get("title", "无标题")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        if snippet:
            formatted.append(f"【{title}】\n{snippet}\n来源：{link}")
    result = "\n\n".join(formatted) if formatted else "未找到相关搜索结果"
    log.info(f"搜索到的内容：{result}")

    # 4. 存入缓存（仅成功时）
    redis_cache.set_search(query, result)
    return result

if __name__ == '__main__':
    query = "今天的时政新闻"
    result = asyncio.run(web_search.ainvoke({"query": query}))
    print(result)