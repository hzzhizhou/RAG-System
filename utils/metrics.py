# utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# HTTP 请求指标
http_requests_total = Counter(
    'web_agent_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'web_agent_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

# Agent 调用指标
agent_duration_seconds = Histogram(
    'web_agent_agent_duration_seconds',
    'Agent invocation duration',
    buckets=(0.5, 1, 2, 4, 8, 15)
)

# 搜索工具指标
# 下面三个Counter用于统计搜索工具的相关情况
# 1. search_calls_total 记录所有被调用的搜索接口的总次数
# 2. search_cache_hits 记录命中缓存的搜索结果数
# 3. search_cache_misses 记录未命中缓存（即需新搜索）的次数
search_calls_total = Counter(
    'web_agent_search_calls_total',           # 指标名称
    'Total search API calls'                  # 指标描述
)
search_cache_hits = Counter(
    'web_agent_search_cache_hits',
    'Search cache hits'
)
search_cache_misses = Counter(
    'web_agent_search_cache_misses',
    'Search cache misses'
)
# 搜索错误计数
search_errors_total = Counter('web_agent_search_errors_total', 'Total search API errors')
# 网页抓取指标
# 下面三个Counter用于统计网页正文抓取工具的使用情况
# 1. fetch_calls_total 记录fetch工具的调用总数
# 2. fetch_cache_hits 记录fetch操作命中缓存的次数
# 3. fetch_cache_misses 记录fetch操作未命中缓存的次数
fetch_calls_total = Counter(
    'web_agent_fetch_calls_total',
    'Total fetch calls'
)
fetch_cache_hits = Counter(
    'web_agent_fetch_cache_hits',
    'Fetch cache hits'
)
fetch_cache_misses = Counter(
    'web_agent_fetch_cache_misses',
    'Fetch cache misses'
)

# 系统信息
# 下面这段代码用于定义/暴露服务运行的静态信息（如版本、环境）
system_info = Info(
    'web_agent_system',                # 指标名称
    'System information'               # 指标描述
)
system_info.info({
    'version': '1.0',                  # 版本号
    'environment': 'production'        # 环境类型
})