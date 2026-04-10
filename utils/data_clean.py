import re
from langchain.schema import Document

# ===================== 第一层：加载后 → 基础清洗（保命：防崩溃、防向量污染）=====================
def clean_documents(docs: list[Document]) -> list[Document]:
    """
    清洗目标：删除致命脏数据，保证文本能正常向量化
    不做此层：Embedding报错、向量漂移、检索全错、LLM乱码
    """
    cleaned = []
    for doc in docs:
        text = doc.page_content

        # 1. 删除空文本
        if not text or len(text.strip()) < 3:
            continue

        # 2. 删除BOM头、非法编码
        text = text.replace("\ufeff", "").replace("\xa0", " ")

        # 3. 删除控制字符（导致向量/LLM崩溃的元凶）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # 4. 统一换行：Windows/Mac/Linux换行归一化
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 5. 多余空行 → 1行
        text = re.sub(r"\n+", "\n", text)

        # 6. 全角空格→半角，多空格→1个
        text = text.replace("　", " ")
        text = re.sub(r" +", " ", text)

        # 7. 删除页眉、页脚、页码、版权、水印
        text = re.sub(r"第\d+页\s*/?\s*共\d+页", "", text)
        text = re.sub(r"版权所有|内部资料|严禁复制|仅供内部使用", "", text)
        text = re.sub(r"文档编号[:：]\w+", "", text)

        # 8. 首尾去空
        text = text.strip()

        # 过滤过短无效内容
        if len(text) < 5:
            continue

        doc.page_content = text
        cleaned.append(doc)
    return cleaned

# ===================== 第二层：分块后 → 分块清洗（保质：保证Chunk语义完整）=====================
def clean_chunks(chunks: list[Document]) -> list[Document]:
    """
    清洗目标：修复分块碎片、去重、过滤无效块
    不做此层：召回碎片、答案残缺、幻觉飙升、向量库臃肿
    """
    cleaned = []
    seen_content = set()

    for chunk in chunks:
        text = chunk.page_content.strip()

        # 1. 过滤超短碎片（无语义）
        if len(text) < 25:
            continue

        # 2. 过滤纯符号/纯数字碎片
        if re.fullmatch(r"^[\d\s\W]+$", text):
            continue

        # 3. 去重（完全重复的Chunk直接丢弃）
        if text in seen_content:
            continue
        seen_content.add(text)

        # 4. 清理切割产生的半截符号
        text = re.sub(r"^[，。、；：？！,.\-:\s]+", "", text)
        text = re.sub(r"[，。、；：？！,.\-:\s]+$", "", text)

        # 5. 最终过滤
        if len(text) < 15:
            continue

        chunk.page_content = text.strip()
        cleaned.append(chunk)
    return cleaned

# ===================== 第三层：检索后 → 上下文清洗（合规：给LLM前最后兜底）=====================
def clean_context(docs: list[Document]) -> list[Document]:
    """
    清洗目标：脱敏敏感信息、去重、格式规整
    不做此层：敏感信息泄露、LLM输出冗余、合规风险
    """
    cleaned = []
    seen = set()

    for doc in docs:
        text = doc.page_content

        # 1. 敏感信息脱敏（企业必备）
        text = re.sub(r"1[3-9]\d{9}", "【手机号脱敏】", text)  # 手机号
        text = re.sub(r"\d{11}|\d{18}", "【证件号脱敏】", text)  # 身份证/银行卡
        text = re.sub(r"\w+@\w+\.\w+", "【邮箱脱敏】", text)  # 邮箱

        # 2. 去重
        if text in seen:
            continue
        seen.add(text)

        # 3. 最终规整
        text = text.strip()
        if len(text) < 10:
            continue

        doc.page_content = text
        cleaned.append(doc)
    return cleaned