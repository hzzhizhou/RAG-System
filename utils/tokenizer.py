"""
企业级中文分词：自定义词典、停用词过滤、容错处理
"""
import jieba
from typing import List
from config.settings import STOP_WORDS
from logs.log_config import log
import re

class ChineseTokenizer:
    """中文分词工具，适配RAG检索场景（支持小写兼容）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_tokenizer()
        return cls._instance

    def _init_tokenizer(self):
        """初始化分词器：加载词典、停用词、优化配置"""
        # 1. 强制添加核心关键词（确保不被拆分）
        jieba.add_word("分块")
        jieba.add_word("rag")  # 小写兼容
        jieba.add_word("RAG")
        jieba.add_word("大模型")
        jieba.add_word("私有化部署")

        # 2. 加载停用词
        self.stop_words = self._load_stop_words()
        # 3. 开启并行分词（性能优化）
        import sys
        if sys.platform != "win32":
            try:
                jieba.enable_parallel(4)
            except Exception as e:
                log.warning(f"并行分词开启失败：{e}")
        else:
            log.info("Windows 系统，跳过并行分词（使用单线程）")
    def _load_stop_words(self) -> set:
        """加载停用词（全异常兜底）"""
        stop_words = set()
        try:
            # 修正原代码bug：open缺少文件路径参数
            with open(STOP_WORDS, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()  # 停用词统一小写
                    if word:
                        stop_words.add(word)
            log.info(f"停用词加载完成，数量：{len(stop_words)}")
        except FileNotFoundError:
            # 企业级兜底停用词
            stop_words = {"的", "是", "在", "和", "有", "了", "我", "你", "他", "\n", "\t", " "}
            log.warning("停用词文件不存在，使用兜底停用词")
        except Exception as e:
            stop_words = {"的", "是", "在", "和", "有"}
            log.error(f"停用词加载异常：{str(e)}")
        return stop_words

    def clean_text(self, text: str) -> str:
        """企业级文本清洗：保留中文/英文/数字，去除所有干扰字符"""
        if not text:
            return ""
        text = text.strip()
        # 正则清洗：只保留有效字符（兼容中英文）
        return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)

    def tokenize(self, text: str) -> List[str]:
        """
        标准分词接口（企业级）
        返回：去重、去停用词、清洗后的关键词列表（统一小写）
        """
        # 1. 文本清洗
        clean_text = self.clean_text(text)
        if not clean_text:
            return []

        # 2. 精准分词（精确模式）
        tokens = jieba.lcut(clean_text, cut_all=False)

        # 3. 过滤规则：去停用词+空字符串，统一转为小写
        filtered_tokens = [
            t.strip().lower() for t in tokens 
            if t.strip() and t.strip().lower() not in self.stop_words
        ]

        # 4. 去重（保留顺序）
        filtered_tokens = list(dict.fromkeys(filtered_tokens))

        return filtered_tokens

# 全局单例分词器
tokenizer = ChineseTokenizer()