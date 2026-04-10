"""
RAG 系统内部测试脚本
功能：直接调用 RetrievalService 和 AnswerGenerator，对预定义问题进行检索和生成
使用方法：确保已运行 data_layer.py 完成入库，然后执行 python test_rag_system.py
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_SEED, DASHSCOPE_API_KEY, SESSION_ID
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from core.retriever_layer import RetrievalService
from core.generation_layer import AnswerGenerator
from core.chat_history_factory import init_chat_history
from utils.thread_pool_manager import init_thread_pools
from langchain_community.chat_models.tongyi import ChatTongyi


# ===================== 测试问题集 =====================
QUESTIONS = [
    # RAG 技术类
    # "企业级 RAG 系统包含哪五大模块？",
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
    # "商务部“出口中国”品牌活动今年确定的主题国包括哪些？",
    # # 科技创新类
    # "春季过敏患者使用“网红止痒神药”可能带来哪些副作用？",
    # "预防百日咳最核心最有效的手段是什么？",
    # "“磐石·禹衡碳核算大模型”首次将哪三个领域纳入统一的全景框架？",
    # "该碳核算大模型的名称“禹衡”有何寓意？",
    # # 食品与消费类
    # "烧烤品牌在食材创新方面围绕哪三个方向进行升级？",
    "2025 年全国烧烤市场规模约为多少亿元？",
    # "联合利华饮食策划为烧烤品牌推出了哪些“一酱多用”的解决方案？",
]


async def test_single_question(retrieval_service, answer_generator, question: str, idx: int, total: int):
    """测试单个问题：检索 + 生成，打印结果"""
    print(f"\n[{idx}/{total}] 问题: {question}")
    print("-" * 60)
    
    start = time.time()
    # 检索（不使用上下文改写，单轮测试）
    docs, retriever_type = await retrieval_service.retrieve(
        question=question,
        route_mode="hybrid",
        use_context=False,
        use_hyde=False,
        use_multi=False
    )
    retrieval_time = time.time() - start
    
    # 打印检索到的文档（取前2个）
    print(f"检索器: {retriever_type} | 耗时: {retrieval_time:.2f}s | 返回文档数: {len(docs)}")
    if docs:
        print("\n检索到的文档片段（最多2个）:")
        for i, doc in enumerate(docs[:2]):
            content_preview = doc.page_content[:200].replace('\n', ' ')
            print(f"  [{i+1}] {content_preview}...")
            print(f"      来源: {doc.metadata.get('file_name', '未知')}")
    
    # 生成答案
    gen_start = time.time()
    answer = await answer_generator.generate(question, docs, session_id=None)
    gen_time = time.time() - gen_start
    
    print(f"\n生成答案 (耗时 {gen_time:.2f}s):")
    print(answer)
    print("-" * 60)


async def main():
    print("=" * 80)
    print("RAG 系统内部测试")
    print("=" * 80)
    
    # 初始化组件
    print("\n初始化组件...")
    init_thread_pools()
    
    # 向量库
    vector_store = ChromaVector()
    
    # 对话历史（测试中不使用，但需要传入）
    chat_history = init_chat_history("test_session")
    
    # LLM
    llm = ChatTongyi(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        api_key=DASHSCOPE_API_KEY,
        streaming=False  # 测试中使用非流式
    )
    
    # 检索服务
    retrieval_service = RetrievalService(vector_store, llm, chat_history)
    print("检索服务初始化完成")
    
    # 生成器
    answer_generator = AnswerGenerator(llm)
    print("生成器初始化完成")
    
    # 开始测试
    total = len(QUESTIONS)
    print(f"\n开始测试 {total} 个问题...\n")
    
    for idx, q in enumerate(QUESTIONS, 1):
        await test_single_question(retrieval_service, answer_generator, q, idx, total)
    
    print("\n" + "=" * 80)
    print("所有测试完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())