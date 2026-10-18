"""统一 LLM 调用客户端 (OpenAI-compatible API)"""
import logging
import json
import os
import threading
import time
import uuid
from typing import List, Dict, Optional

import openai

logger = logging.getLogger(__name__)
_raw_log_lock = threading.Lock()


class _RawHTTPClient(openai.DefaultHttpxClient):
    """Optional experiment log below SDK retries, including failed attempts."""
    def __init__(self, raw_log):
        super().__init__()
        self.raw_log = raw_log

    def _write(self, record):
        with _raw_log_lock:
            with open(self.raw_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def send(self, request, *args, **kwargs):
        identity = uuid.uuid4().hex
        context = dict(id=identity, thread=threading.get_ident(), url=str(request.url))
        self._write(dict(context, event="request", time=time.time(),
                         body=request.content.decode("utf-8"),
                         timeout=request.extensions.get("timeout")))
        try:
            response = super().send(request, *args, **kwargs)
            response.read()
        except Exception as exc:
            self._write(dict(context, event="error", time=time.time(),
                             error=repr(exc)))
            raise
        self._write(dict(context, event="response", time=time.time(),
                         status_code=response.status_code, body=response.text))
        return response


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
        self.raw_log = os.environ.get("EGREFINE_RAW_LOG")
        self.log_context = threading.local()
        http_log = os.environ.get("EGREFINE_HTTP_RAW_LOG")
        client_options = {"http_client": _RawHTTPClient(http_log)} if http_log else {}
        self._client = openai.OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=self.timeout,
            **client_options,
        )

    def chat(self, messages: List[Dict[str, str]], *, trace=None,
             temperature: Optional[float] = None) -> str:
        """发送 chat completion 请求，返回 assistant 回复文本。

        内置 retry 逻辑，失败时指数退避重试。

        Args:
            temperature: per-call override. Callers that need a different
                sampling temperature must pass it here instead of mutating
                ``self.temperature``: a client instance is shared by many
                worker threads, so mutating it leaks the value into
                concurrent calls (see experiment_records/2026-10-16-*).
        """
        last_exc = None
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        for attempt in range(self.max_retries):
            event = {"request": kwargs, "attempt": attempt + 1}
            if trace is not None:
                trace.append(event)
            try:
                response = self._client.chat.completions.create(**kwargs)
                if trace is not None:
                    event["response"] = response.model_dump(mode="json")
                if self.raw_log:
                    record = {"time": time.time(), "endpoint": str(self._client.base_url),
                              "context": getattr(self.log_context, "record", {}),
                              "request": kwargs, "response": response.model_dump(mode="json")}
                    with _raw_log_lock:
                        with open(self.raw_log, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return response.choices[0].message.content
            except openai.BadRequestError as e:
                if trace is not None:
                    event["error"] = str(e)
                # 400 errors (e.g. prompt too long) are not retryable
                logger.warning("LLM call got 400 BadRequest (not retryable): %s", e)
                return ""
            except Exception as e:
                if trace is not None:
                    event["error"] = str(e)
                last_exc = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1, self.max_retries, e, wait,
                    )
                    time.sleep(wait)
        raise last_exc
