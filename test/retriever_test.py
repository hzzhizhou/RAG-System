"""
检索精度测试：混合检索 vs 混合检索+重排序
评估指标：Precision@K, Recall@K, MRR
使用方法：
1. 确保已运行 data_layer.py 入库
2. python test_retrieval_accuracy.py
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Set

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import RERANK_TOP_N
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from core.retriever_layer import RetrievalService
from core.chat_history_factory import init_chat_history
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.thread_pool_manager import init_thread_pools


# ===================== 测试问题及期望相关的文档名（根据你的数据集标注） =====================
TEST_CASES = [
    {
        "question": "RAG 的全称是什么？核心思想是什么？",
        "expected_files": ["question_ans.txt", "long_rag_test.txt"]
    },
    {
        "question": "企业级 RAG 系统包含哪五大模块？",
        "expected_files": ["long_rag_test.txt", "question_ans.txt"]
    },
    {
        "question": "什么是多路召回？它的主要作用是什么？",
        "expected_files": ["long_rag_test.txt", "rag_test.txt"]
    },
    {
        "question": "检索参数 top_k 一般设置为多少？为什么不能太大或太小？",
        "expected_files": ["long_rag_test.txt", "question_ans.txt"]
    },
    {
        "question": "重排序（Rerank）能带来多少精确率提升？",
        "expected_files": ["long_rag_test.txt", "rag_test.txt"]
    },
    {
        "question": "中方暂停对美稀土出口管制措施至哪一天？",
        "expected_files": ["财经.docx"]
    },
    {
        "question": "预防百日咳最核心最有效的手段是什么？",
        "expected_files": ["科创.docx"]
    },
    {
        "question": "“磐石·禹衡碳核算大模型”首次将哪三个领域纳入统一的全景框架？",
        "expected_files": ["时政.docx"]
    },
    {
        "question": "2025 年全国烧烤市场规模约为多少亿元？",
        "expected_files": ["食品.docx"]
    },
]


def precision_at_k(retrieved_files: List[str], expected_set: Set[str], k: int) -> float:
    """Precision@K：前K个结果中相关文档的比例"""
    if k == 0:
        return 0.0
    top_k = retrieved_files[:k]
    relevant = sum(1 for f in top_k if f in expected_set)
    return relevant / k


def recall_at_k(retrieved_files: List[str], expected_set: Set[str], k: int) -> float:
    """Recall@K：前K个结果中相关文档占所有相关文档的比例"""
    total_expected = len(expected_set)
    if total_expected == 0:
        return 0.0
    top_k = retrieved_files[:k]
    relevant = sum(1 for f in top_k if f in expected_set)
    return relevant / total_expected


def mrr(retrieved_files: List[str], expected_set: Set[str]) -> float:
    """MRR：第一个相关文档的倒数排名（单个问题）"""
    for rank, f in enumerate(retrieved_files, start=1):
        if f in expected_set:
            return 1.0 / rank
    return 0.0


async def retrieve_without_rerank(retrieval_service, question: str, k: int = 5):
    """仅混合检索（RRF融合），不重排序"""
    docs = await retrieval_service.hybrid_retriever.ainvoke(question)
    docs = docs[:k]
    file_names = []
    for doc in docs:
        name = doc.metadata.get("file_name", "unknown")
        if "/" in name or "\\" in name:
            name = Path(name).name
        file_names.append(name)
    return file_names


async def retrieve_with_rerank(retrieval_service, question: str, k: int = 5):
    """混合检索 + 重排序（使用已有的 ScoreReranker）"""
    # 注意：retrieve 方法内部已经重排序并返回 RERANK_TOP_N 个文档
    # 我们直接用它，然后取前k个（如果k <= RERANK_TOP_N）
    docs, _ = await retrieval_service.retrieve(
        question=question,
        route_mode="hybrid",
        use_context=False,
        use_hyde=False,
        use_multi=False
    )
    docs = docs[:k]
    file_names = []
    for doc in docs:
        name = doc.metadata.get("file_name", "unknown")
        if "/" in name or "\\" in name:
            name = Path(name).name
        file_names.append(name)
    return file_names


async def evaluate_retrieval(retrieval_service, test_cases: List[Dict], k_values: List[int] = [1, 3, 5]):
    """评估检索精度"""
    results = {
        "without_rerank": {"precision": {k: [] for k in k_values}, "recall": {k: [] for k in k_values}, "mrr": []},
        "with_rerank": {"precision": {k: [] for k in k_values}, "recall": {k: [] for k in k_values}, "mrr": []}
    }
    
    for case in test_cases:
        q = case["question"]
        expected_set = set(case["expected_files"])
        
        # 不带重排序
        retrieved_files = await retrieve_without_rerank(retrieval_service, q, max(k_values))
        for k in k_values:
            p = precision_at_k(retrieved_files, expected_set, k)
            r = recall_at_k(retrieved_files, expected_set, k)
            results["without_rerank"]["precision"][k].append(p)
            results["without_rerank"]["recall"][k].append(r)
        results["without_rerank"]["mrr"].append(mrr(retrieved_files, expected_set))
        
        # 带重排序
        retrieved_files = await retrieve_with_rerank(retrieval_service, q, max(k_values))
        for k in k_values:
            p = precision_at_k(retrieved_files, expected_set, k)
            r = recall_at_k(retrieved_files, expected_set, k)
            results["with_rerank"]["precision"][k].append(p)
            results["with_rerank"]["recall"][k].append(r)
        results["with_rerank"]["mrr"].append(mrr(retrieved_files, expected_set))
    
    # 计算平均值
    avg_results = {}
    for mode in ["without_rerank", "with_rerank"]:
        avg_results[mode] = {}
        for k in k_values:
            avg_results[mode][f"Precision@{k}"] = sum(results[mode]["precision"][k]) / len(results[mode]["precision"][k])
            avg_results[mode][f"Recall@{k}"] = sum(results[mode]["recall"][k]) / len(results[mode]["recall"][k])
        avg_results[mode]["MRR"] = sum(results[mode]["mrr"]) / len(results[mode]["mrr"])
    
    return avg_results


async def main():
    print("=" * 80)
    print("检索精度测试：混合检索 (RRF) vs 混合检索 + 重排序 (ScoreReranker)")
    print("=" * 80)
    
    # 初始化组件
    init_thread_pools()
    vector_store = ChromaVector()
    # 创建一个假的 LLM（检索不需要，但 RetrievalService 需要）
    from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_SEED, DASHSCOPE_API_KEY
    llm = ChatTongyi(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        api_key=DASHSCOPE_API_KEY,
        streaming=False
    )
    chat_history = init_chat_history("test_session")
    retrieval_service = RetrievalService(vector_store, llm, chat_history)
    
    print(f"测试问题数量: {len(TEST_CASES)}")
    print(f"重排序 top_n 设置: {RERANK_TOP_N}")
    print("-" * 80)
    
    # 评估
    results = await evaluate_retrieval(retrieval_service, TEST_CASES, k_values=[1, 3, 5])
    
    # 输出结果
    print("\n【不带重排序】")
    for metric, value in results["without_rerank"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n【带重排序】")
    for metric, value in results["with_rerank"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("提升效果:")
    for metric in results["without_rerank"]:
        old = results["without_rerank"][metric]
        new = results["with_rerank"][metric]
        delta = new - old
        pct = (delta / old * 100) if old != 0 else float('inf')
        print(f"  {metric}: {old:.4f} → {new:.4f} ({delta:+.4f}, {pct:+.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())