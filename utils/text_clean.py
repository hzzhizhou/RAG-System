import re
import unicodedata
# 在 TextCleaner 类定义之后，添加以下函数

from langchain_core.documents import Document
from typing import List
#------------------Document对象类型数据清洗------
def clean_documents(docs: List[Document]) -> List[Document]:
    """
    对文档列表中的每个文档进行文本清洗
    :param docs: 原始 Document 列表
    :return: 清洗后的 Document 列表（新列表，原文档不变）
    """
    cleaned = []
    for doc in docs:
        # 清洗文本内容
        cleaned_content = TextCleaner.clean_text(doc.page_content)
        # 创建新文档，保留原元数据（复制避免引用）
        new_doc = Document(
            page_content=cleaned_content,
            metadata=doc.metadata.copy()
        )
        cleaned.append(new_doc)
    return cleaned

#----------------------字符串数据清洗--------------------------
class TextCleaner:
    """基础文本清洗工具类"""

    @staticmethod
    def remove_control_chars(text):
        """移除 ASCII 控制字符（保留换行、回车、制表符）"""
        # 移除除了 \n, \r, \t 以外的控制字符
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    @staticmethod
    def normalize_whitespace(text) :
        """归一化空白：将连续的空白字符（空格、换行、制表等）替换为单个空格，并去除首尾空白"""
        # 先将各种空白字符（包括多个空格、换行、制表等）统一为空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # @staticmethod
    # def to_simplified(text: str) -> str:
    #     """
    #     繁体转简体（可选，依赖zhconv）
    #     若未安装zhconv，则返回原文本并打印警告
    #     """
    #     try:
    #         import zhconv
    #         return zhconv.convert(text, 'zh-cn')
    #     except ImportError:
    #         # 如果未安装，可记录日志，简单返回原文本
    #         # 在实际项目中，建议安装：pip install zhconv
    #         return text

    @staticmethod
    def fullwidth_to_halfwidth(text) :
        """全角字符转半角字符"""
        result = []
        for char in text:
            code = ord(char)
            # 全角字母、数字、符号范围：65281～65374（对应半角33～126）
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:  # 全角空格
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def remove_bom(text):
        """移除 UTF-8 BOM 头"""
        if text.startswith('\ufeff'):
            return text[1:]
        return text

    @staticmethod
    def clean_text(text) :
        """执行完整的文本清洗（顺序：BOM移除 -> 控制字符 -> 全角转半角 -> 繁转简 -> 空白归一化）"""
        text = TextCleaner.remove_bom(text)
        text = TextCleaner.remove_control_chars(text)
        text = TextCleaner.fullwidth_to_halfwidth(text)
        # text = TextCleaner.to_simplified(text)
        text = TextCleaner.normalize_whitespace(text)
        return text