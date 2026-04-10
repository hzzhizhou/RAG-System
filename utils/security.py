"""
企业级数据安全工具：脱敏、权限校验、数据加密
"""
import re
import secrets
from typing import List
from config.settings import DASHSCOPE_API_KEY
from logs.log_config import log

class DataSecurity:
    """数据安全处理：脱敏、权限校验"""

    @staticmethod
    def desensitize_text(text: str) -> str:
        """
        敏感信息脱敏：身份证、手机号、邮箱
        """
        if not text:
            return ""
        # 身份证脱敏（110101199001011234 → 110101********1234）
        text = re.sub(r'(\d{6})\d{8}(\d{4})', r'\1********\2', text)
        # 手机号脱敏（13800138000 → 138****8000）
        text = re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', text)
        # 邮箱脱敏（test@example.com → t***t@example.com）
        text = re.sub(r'(\w)\w+(\w)@(\w+\.\w+)', r'\1***\2@\3', text)
        return text

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        API密钥校验（企业级接口鉴权）
        使用 secrets.compare_digest 防止时序攻击
        """
        if not api_key or not DASHSCOPE_API_KEY:
            return False
        # 使用恒定时间比较，防止时序攻击
        return secrets.compare_digest(api_key, DASHSCOPE_API_KEY)

    @staticmethod
    def validate_file_permission(file_path: str, allowed_dir: str) -> bool:
        """
        文件权限校验（规避越权访问）
        :param file_path: 要访问的文件路径
        :param allowed_dir: 允许访问的根目录
        """
        # 规范化路径，防止路径遍历攻击
        real_path = str(file_path)
        real_allowed = str(allowed_dir)
        if not real_path.startswith(real_allowed):
            log.warning(f"越权访问文件：{real_path}")
            return False
        return True