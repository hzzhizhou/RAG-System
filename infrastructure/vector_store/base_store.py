from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from langchain_core.documents import Document

class BaseVectorStore(ABC):
    """向量数据库抽象接口（同步 + 异步）"""

    # ========== 同步方法（供本地/同步场景使用） ==========
    @abstractmethod
    def add_documents(self, docs: List[Document], batch_size: int = 100) -> None:
        """同步批量添加文档"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Document]:
        """同步相似度检索"""
        pass

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Tuple[Document, float]]:
        """同步带分数检索"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """同步清空向量库"""
        pass

    @abstractmethod
    def _get_all_documents(self) -> List[Document]:
        """获取所有文档（内部使用）"""
        pass

    @abstractmethod
    def as_retriever(self, **kwargs):
        """返回检索器（通常同步）"""
        pass

    # ========== 异步方法（供高并发API使用） ==========
    @abstractmethod
    async def aadd_documents(self, docs: List[Document], batch_size: int = 100) -> None:
        pass

    @abstractmethod
    async def asimilarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Document]:
        pass

    @abstractmethod
    async def asimilarity_search_with_score(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    async def aclear(self) -> None:
        pass

    @abstractmethod
    async def delete_by_file(self, file_path: str):
        pass