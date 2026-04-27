"""统一 LLM 调用客户端 (OpenAI-compatible API)"""
import logging
import time
from typing import List, Dict

import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """封装 OpenAI-compatible chat completion 调用。"""

    def __init__(self, config: dict):
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        self.max_retries = config.get("max_retries", 3)
        self.extra_body = config.get("extra_body")
        # openai-python default timeout is 600s; reasoning models with long
        # CoT on complex schemas can easily exceed it. Allow override via
        # config.timeout (seconds).
        self.timeout = config.get("timeout", 600)
        self._client = openai.OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=self.timeout,
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """发送 chat completion 请求，返回 assistant 回复文本。

        内置 retry 逻辑，失败时指数退避重试。
        """
        last_exc = None
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except openai.BadRequestError as e:
                # 400 errors (e.g. prompt too long) are not retryable
                logger.warning("LLM call got 400 BadRequest (not retryable): %s", e)
                return ""
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1, self.max_retries, e, wait,
                    )
                    time.sleep(wait)
        raise last_exc
