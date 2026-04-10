
from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import BaseMessage


class BaseChatHistory(ABC):
    @abstractmethod
    def messages(self)->List[BaseMessage]:
        pass
    @abstractmethod
    def add_message(self,messages:BaseMessage):
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def rewrite_question(self, question: str, llm) -> str:
        pass