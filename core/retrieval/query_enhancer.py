from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from logs.log_config import log

class QueryEnhancer:
    def __init__(self, llm):
        self.llm = llm
        self.multi_query_prompt = ChatPromptTemplate.from_messages([
            ("system", "请为以下问题生成 {n} 个不同的搜索查询，用逗号分隔，每个查询应覆盖不同的角度。"),
            ("human", "问题：{question}\n查询：")
        ])
        self.synonym_prompt = ChatPromptTemplate.from_messages([
            ("system", "请为以下查询中的关键词生成同义词或相关词，返回扩展后的查询。"),
            ("human", "查询：{query}\n扩展后：")
        ])
        self.hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", "请根据以下问题，生成一段可能出现在相关文档中的文本（假设文档）。该文档应详细、专业，能够用于语义检索。"),
            ("human", "问题：{question}\n假设文档：")
        ])

    async def hyde(self, question: str) -> str:
        chain = self.hyde_prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"question": question})
        except Exception as e:
            log.error(f"HyDE 生成失败: {e}")
            return question

    async def multi_query(self, question: str, n: int = 3) -> List[str]:
        chain = self.multi_query_prompt | self.llm | StrOutputParser()
        try:
            result = chain.invoke({"question": question, "n": n})
            queries = [q.strip() for q in result.split(",")]
            return queries[:n]
        except Exception as e:
            log.error(f"多查询生成失败: {e}")
            return [question]

    def synonym_expand(self, query: str) -> str:
        chain = self.synonym_prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"query": query})
        except Exception:
            return query