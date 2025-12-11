"""
通义千问 (Qwen) 客户端实现。
使用阿里云 DashScope API。
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

from .base import BaseLLMClient
from .exceptions import APIError

load_dotenv()

logger = logging.getLogger(__name__)


class QwenClient(BaseLLMClient):
    """通义千问 API 客户端实现。"""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a professional programming assistant. "
        "You **MUST** carefully follow the user's requests."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        max_retries: int = 10,
        retry_delay: int = 60,
        **kwargs
    ):
        """
        初始化通义千问客户端。
        
        Args:
            api_key: DashScope API Key。如果为None，将从环境变量获取
            model: 模型名称 (默认: qwen-plus，可选: qwen-turbo, qwen-plus, qwen-max等)
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            **kwargs: 其他配置参数
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not DASHSCOPE_AVAILABLE:
            raise ImportError(
                "dashscope is not installed. Install with: pip install dashscope"
            )

        if not self.api_key:
            raise APIError("DASHSCOPE_API_KEY not found in environment variables")

        # 设置 API Key
        dashscope.api_key = self.api_key

    def request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        **kwargs
    ) -> Optional[str]:
        """
        发送请求到通义千问 API。
        
        Args:
            prompt: 用户提示
            system_prompt: 可选的系统提示
            temperature: 采样温度
            max_tokens: 最大token数
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数
            
        Returns:
            响应文本，失败则返回None
        """
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = Generation.call(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    result_format='message',  # 返回消息格式
                    **kwargs
                )

                if response.status_code == 200:
                    return response.output.choices[0].message.content
                else:
                    error_msg = f"API returned status {response.status_code}: {response.message}"
                    logger.warning("Attempt %d failed: %s", attempt, error_msg)
                    
                    # 如果是限流错误，等待更长时间
                    if response.status_code == 429:
                        wait_time = self.retry_delay * attempt
                        logger.info("Rate limited. Waiting %d seconds...", wait_time)
                        time.sleep(wait_time)
                    elif attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                    else:
                        raise APIError(error_msg)

            except Exception as e:
                logger.warning("Attempt %d failed: %s", attempt, e)

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries reached. Request failed.")
                    raise APIError(f'API request failed after {self.max_retries} attempts: {e}') from e

        return None

    def is_available(self) -> bool:
        """检查通义千问 API 是否可用。"""
        try:
            response = Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("Qwen API availability check failed: %s", e)
            return False

