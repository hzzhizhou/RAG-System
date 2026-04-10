import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict

# 配置
API_URL = "http://localhost:8000/rag/query"  # 修改为您的 API 地址
CONCURRENT_USERS = 5          # 模拟并发用户数
TOTAL_REQUESTS = 10            # 总请求数（每个用户可能发多个，实际按总量控制）
REQUEST_INTERVAL = 0.1          # 每个用户连续请求的间隔（秒），避免过载
QUERY_TEMPLATE = "什么是RAG？"    # 可以是固定问题，也可以从列表随机取

# 可选：准备一批不同的问题
QUERIES = [
    "什么是RAG？",
    "BM25算法的工作原理",
    "向量检索和关键词检索的区别",
    "混合检索如何配置？"
]

async def send_request(session: aiohttp.ClientSession, question: str, request_id: int) -> Dict:
    """发送单个请求并记录耗时"""
    start = time.perf_counter()
    try:
        payload = {"question": question, "session_id": f"test_{request_id}"}
        async with session.post(API_URL, json=payload) as resp:
            text = await resp.text()
            status = resp.status
            end = time.perf_counter()
            return {
                "request_id": request_id,
                "success": status == 200,
                "status_code": status,
                "duration": end - start,
                "response": text[:200]  # 仅保留前200字符用于日志
            }
    except Exception as e:
        end = time.perf_counter()
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
            "duration": end - start
        }

async def user_worker(session: aiohttp.ClientSession, user_id: int, num_requests: int, results: List):
    """模拟一个用户连续发送多个请求（带间隔）"""
    for i in range(num_requests):
        # 随机选择问题（可从列表中取，也可以固定）
        question = QUERIES[user_id % len(QUERIES)] if QUERIES else QUERY_TEMPLATE
        request_id = user_id * 100 + i
        result = await send_request(session, question, request_id)
        results.append(result)
        if REQUEST_INTERVAL > 0:
            await asyncio.sleep(REQUEST_INTERVAL)

async def run_concurrent_test():
    """主测试函数"""
    # 计算每个用户应发送的请求数（平均分配）
    requests_per_user = TOTAL_REQUESTS // CONCURRENT_USERS
    remainder = TOTAL_REQUESTS % CONCURRENT_USERS

    print(f"开始高并发测试: 并发用户数={CONCURRENT_USERS}, 总请求数={TOTAL_REQUESTS}")
    print(f"每个用户请求数: {requests_per_user} (余数 {remainder} 分配给前几个用户)")
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []
        results = []
        for u in range(CONCURRENT_USERS):
            num = requests_per_user + (1 if u < remainder else 0)
            tasks.append(user_worker(session, u, num, results))
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    # 统计结果
    success_list = [r for r in results if r.get("success")]
    failed_list = [r for r in results if not r.get("success")]
    durations = [r["duration"] for r in success_list]
    if durations:
        avg = statistics.mean(durations)
        p50 = statistics.median(durations)
        p90 = statistics.quantiles(durations, n=10)[8] if len(durations) >= 10 else max(durations)
        p99 = statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations)
        min_t = min(durations)
        max_t = max(durations)
    else:
        avg = p50 = p90 = p99 = min_t = max_t = 0

    print("\n========== 测试结果 ==========")
    print(f"总耗时: {total_time:.2f}s")
    print(f"总请求数: {len(results)}")
    print(f"成功: {len(success_list)}")
    print(f"失败: {len(failed_list)}")
    print(f"QPS (吞吐量): {len(success_list) / total_time:.2f} req/s")
    print("\n延迟统计 (仅成功请求):")
    print(f"  平均: {avg*1000:.1f} ms")
    print(f"  中位数 (P50): {p50*1000:.1f} ms")
    print(f"  P90: {p90*1000:.1f} ms")
    print(f"  P99: {p99*1000:.1f} ms")
    print(f"  最小: {min_t*1000:.1f} ms")
    print(f"  最大: {max_t*1000:.1f} ms")

    # 打印失败示例
    if failed_list:
        print("\n失败请求示例:")
        for f in failed_list[:5]:
            print(f"  request_id={f['request_id']}, error={f.get('error', f.get('status_code'))}")

if __name__ == "__main__":
    asyncio.run(run_concurrent_test())