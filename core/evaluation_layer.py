"""
评估层：RAGAS评估+稳定化+指标监控
规避：评估结果波动、指标无监控问题
"""
import time
from utils.thread_pool_manager import init_thread_pools
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall
)
from datasets import Dataset
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from core.retriever_layer import RetrievalService
from core.generation_layer import AnswerGenerator
from langchain_community.chat_models.tongyi import ChatTongyi
from core.chat_history_factory import init_chat_history
from config.settings import (
    EVAL_RUNS, EVAL_THRESHOLDS, LLM_MODEL, LLM_TEMPERATURE, LLM_SEED,
    LOCAL_EMBEDDING_MODEL
)
from logs.log_config import evaluation_layer_log as log


class RAGEvaluator:
    """企业级RAG评估器：稳定化指标、监控告警、历史记录"""

    def __init__(self, vector_store, retrieval_service, answer_generator):
        self.vector_store = vector_store
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator

        # 初始化 RAGAS 评估模型（固定参数，保证结果稳定）
        self.ragas_llm = ChatTongyi(model="qwen-plus", temperature=0.01,top_p = 0.01, seed=0)

        # 获取嵌入模型（用于 answer_relevancy）
        embedding_model = vector_store.embedding_model
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    async def _generate_test_data_async(self,
                                        questions: List[str],
                                        ground_truths: Optional[List[str]] = None) -> Dataset:
        """
        异步生成评估所需的数据集（并行执行检索和生成）
        """
        async def process_one(q: str, gt: Optional[str] = None):
            # 执行检索（注意：retrieve 是异步方法）
            docs, retriever_type = await self.retrieval_service.retrieve(
                question=q,
                use_context=False,   # 评估时不使用对话历史改写
                use_hyde=False,
                use_multi=False
            )
            # 生成答案
            answer = await self.answer_generator.generate(q, docs, session_id=None)
            contexts = [doc.page_content for doc in docs]
            return {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": gt if gt else ""
            }

        tasks = []
        for i, q in enumerate(questions):
            gt = ground_truths[i] if ground_truths else None
            tasks.append(process_one(q, gt))

        results = await asyncio.gather(*tasks)

        data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
        }
        if ground_truths:
            data["ground_truth"] = [r["ground_truth"] for r in results]

        return Dataset.from_dict(data)

    # 在 evaluation_layer.py 中替换 evaluate_async 方法
    async def evaluate_async(self,
                            questions: List[str],
                            ground_truths: Optional[List[str]] = None,
                            runs: int = None) -> Dict[str, float]:
        if runs is None:
            runs = EVAL_RUNS
        # 生成数据集
        dataset = await self._generate_test_data_async(questions, ground_truths)
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics.append(context_recall)
        # 只运行一次（多次无意义）
        if runs != 1:
            log.warning(f"RAGAS 评估建议 runs=1，当前 runs={runs}，将只运行一次")
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                raise_exceptions=True,
            )
            # 提取指标
            if hasattr(results, 'to_pandas'):
                df = results.to_pandas()
                # 仅保留数值列
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                avg_results = {col: round(df[col].iloc[0], 4) for col in numeric_cols}
            elif hasattr(results, 'scores'):
                avg_results = {k: round(v, 4) for k, v in results.scores.items()}
            else:
                # 尝试通过 __dict__ 获取浮点数属性
                avg_results = {}
                for k, v in results.__dict__.items():
                    if isinstance(v, float) and not k.startswith('_'):
                        avg_results[k] = round(v, 4)
            log.info(f"评估完成，结果：{avg_results}")
            return avg_results
        except Exception as e:
            log.error(f"评估失败：{e}", exc_info=True)
            return {}

    def evaluate(self,
                 questions: List[str],
                 ground_truths: Optional[List[str]] = None,
                 runs: int = None) -> Dict[str, float]:
        """
        同步包装器（供非异步环境调用）
        """
        return asyncio.run(self.evaluate_async(questions, ground_truths, runs))

    def evaluate_with_report(self,
                             questions: List[str],
                             ground_truths: Optional[List[str]] = None,
                             runs: int = None) -> Dict[str, Any]:
        """
        执行评估并生成详细报告（同步）
        """
        avg_results = self.evaluate(questions, ground_truths, runs)

        report = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_runs": runs if runs else EVAL_RUNS,
            "num_questions": len(questions),
            "metrics": avg_results,
            "thresholds": EVAL_THRESHOLDS,
            "alerts": []
        }

        for metric, threshold in EVAL_THRESHOLDS.items():
            if avg_results.get(metric, 0) < threshold:
                report["alerts"].append({
                    "metric": metric,
                    "value": avg_results[metric],
                    "threshold": threshold,
                    "suggestion": f"建议优化{metric}相关组件"
                })
        return report
    
if __name__ == '__main__':


    async def evaluate_with_ground_truth():
        print("开始初始化组件...")
        vector_store = ChromaVector()
        llm = ChatTongyi(model="qwen3-max", temperature=0.01, top_p=0.1, seed=42)
        init_thread_pools()
        chat_history = init_chat_history("text1")
        retrieval_service = RetrievalService(vector_store, llm, chat_history)
        answer_generator = AnswerGenerator(llm)
        evaluator = RAGEvaluator(vector_store, retrieval_service, answer_generator)

        # 使用单个问题快速测试（减少 token）
        questions = ["RAG 的全称是什么？核心思想是什么？"]
        ground_truths = ["全称：Retrieval-Augmented Generation，检索增强生成。核心思想：不修改大模型权重，在生成回答前从外部知识库检索相关上下文，将可靠信息注入 Prompt。"]

        print("开始评估，请耐心等待（可能需要 1-2 分钟）...")
        start = time.time()
        metrics = await evaluator.evaluate_async(questions, ground_truths=ground_truths, runs=1)
        elapsed = time.time() - start
        print(f"评估完成，耗时 {elapsed:.1f} 秒")
        print("完整指标:", metrics)
        return metrics   # 添加返回值
    # 运行异步函数并打印结果
    result = asyncio.run(evaluate_with_ground_truth())
    print("最终返回结果:", result)