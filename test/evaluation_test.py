"""
RAGAS 评估脚本（20个问题）
基于已有的 RAGEvaluator 进行自动化评估
运行前请确保已运行 data_layer.py 入库所有文档
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_SEED, DASHSCOPE_API_KEY
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from core.retriever_layer import RetrievalService
from core.generation_layer import AnswerGenerator
from core.chat_history_factory import init_chat_history
from core.evaluation_layer import RAGEvaluator
from utils.thread_pool_manager import init_thread_pools
from langchain_community.chat_models.tongyi import ChatTongyi


# ===================== 20个测试问题及标准答案 =====================
QUESTIONS = [
    # RAG 技术类
    "企业级 RAG 系统包含哪五大模块？",
    # "什么是多路召回？它的主要作用是什么？",
    # "检索参数 top_k 一般设置为多少？为什么不能太大或太小？",
    # "RAG 系统出现“长文本中间信息丢失”的常见原因是什么？",
    # "文中总结的三条真实场景核心经验是什么？",
    # "向量数据库 Chroma 有哪些核心特性？",
    # "重排序（Rerank）能带来多少精确率提升？",
    # "简述 RAG 技术的完整流程（索引→检索→生成）。",
    # "通用场景推荐的 chunk_size 和 chunk_overlap 区间是多少？",
    # "RAG 效果 70% 取决于什么？",
    # # 经济与政策类
    # "中方暂停对美稀土出口管制措施至哪一天？",
    # "欧盟近期推出的哪两项法案影响了中欧企业合作的信心？",
    # "商务部“出口中国”品牌活动今年确定的主题国包括哪些？（至少说出三个）",
    # # 科技创新类
    # "春季过敏患者使用“网红止痒神药”可能带来哪些副作用？",
    # "预防百日咳最核心最有效的手段是什么？",
    # "“磐石·禹衡碳核算大模型”首次将哪三个领域纳入统一的全景框架？",
    # "该碳核算大模型的名称“禹衡”有何寓意？",
    # # 食品与消费类
    # "烧烤品牌在食材创新方面围绕哪三个方向进行升级？",
    # "2025 年全国烧烤市场规模约为多少亿元？",
    # "联合利华饮食策划为烧烤品牌推出了哪些“一酱多用”的解决方案？（至少说出两个）",
]

GROUND_TRUTHS = [
    "文档接入层、文本处理层、向量存储层、检索层、生成层。",
    # "多路召回：同时使用向量检索 + BM25 全文检索等多种方式召回。作用：显著提升召回率，尤其在专业术语多、关键词稀疏的场景。",
    # "通常设置为 3~5。过小容易遗漏关键信息；过大引入冗余内容，增加生成耗时并降低答案质量。",
    # "文档过长时，中间段落语义较弱，分块后向量相似度偏低，检索只能命中开头或结尾，中间信息无法召回。",
    # "1. RAG 效果 70% 取决于文本处理与分块，而非模型。2. 没有万能参数，必须按业务场景调优。3. 生产系统必须具备可观测性。",
    # "轻量级，无需复杂部署；支持元数据过滤；内置多种嵌入模型。",
    # "可提升精确率 20% 以上。",
    # "1. 文档索引：知识库切分段落，嵌入向量存入向量库；2. 检索：用户问题向量化，检索相似文档片段；3. 生成：将上下文与问题输入 LLM 生成答案。",
    # "chunk_size：500~800，chunk_overlap：50~100。",
    # "RAG 效果 70% 取决于文本处理与分块，而非模型。",
    # "2026年11月10日。",
    # "《网络安全法》修订草案和《工业加速器法案》。",
    # "英国、西班牙、哈萨克斯坦、肯尼亚、泰国。",
    # "色素沉着、红血丝密布等。",
    # "接种疫苗。",
    # "生产端、消费端、自然源。",
    # "“禹”取自大禹治水，寓意系统治理的智慧；“衡”源自北斗玉衡星，象征成为碳排放核算的时空标尺。",
    # "“鲜”（品质升级）、“新”（地域/小众食材）、“奇”（创意搭配）。",
    # "2680亿元。",
    # "香辣涮肚汤底、金银蒜捞拌酱、辣香干撒粉等。",
]


async def main():
    print("=" * 80)
    print("RAGAS 评估 开始")
    print("=" * 80)

    # 初始化组件
    print("\n初始化组件...")
    init_thread_pools()
    vector_store = ChromaVector()
    llm = ChatTongyi(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        api_key=DASHSCOPE_API_KEY,
        streaming=False
    )
    chat_history = init_chat_history("eval_session")
    retrieval_service = RetrievalService(vector_store, llm, chat_history)
    answer_generator = AnswerGenerator(llm)
    evaluator = RAGEvaluator(vector_store, retrieval_service, answer_generator)

    print(f"向量库持久化目录: {vector_store.persist_dir}")
    print(f"LLM 模型: {LLM_MODEL}")
    print(f"评估问题数量: {len(QUESTIONS)}")

    # 开始评估
    print("\n开始评估，这可能需要几分钟（每个问题都会调用 LLM 生成答案）...")
    start_time = time.time()
    metrics = await evaluator.evaluate_async(
        questions=QUESTIONS,
        ground_truths=GROUND_TRUTHS,
        runs=1
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"评估完成，耗时 {elapsed:.1f} 秒")
    print("RAGAS 指标结果：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 80)

    # 输出告警（如果低于阈值）
    from config.settings import EVAL_THRESHOLDS
    print("\n阈值对比：")
    for metric, threshold in EVAL_THRESHOLDS.items():
        value = metrics.get(metric, 0)
        status = "✅ 达标" if value >= threshold else "⚠️ 低于阈值"
        print(f"  {metric}: {value:.4f} (阈值 {threshold}) {status}")


if __name__ == "__main__":
    asyncio.run(main())