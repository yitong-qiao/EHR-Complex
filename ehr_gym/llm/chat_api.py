import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional
import openai
from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
from ehr_gym.agent.parser import normalize_action_text


class RetryError(Exception):
    """Raised when all API retries are exhausted."""
    pass


def handle_error(e: openai.OpenAIError, itr: int, min_wait: float, max_retry: int) -> str:
    """Raise non-retryable API errors; otherwise wait and return the error category."""
    if isinstance(e, openai.AuthenticationError):
        raise e
    if isinstance(e, openai.BadRequestError):
        raise e
    if isinstance(e, openai.PermissionDeniedError):
        raise e
    if isinstance(e, openai.NotFoundError):
        raise e

    if isinstance(e, openai.RateLimitError):
        wait = min(10 * (2 ** itr), 120) + random.uniform(0, 5)
        time.sleep(wait)
        return "rate_limit"

    if isinstance(e, openai.InternalServerError):
        time.sleep(10)
        return "server_error"

    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        time.sleep(10)
        return "connection_error"

    time.sleep(10)
    return "unknown_error"

def make_system_message(content: str) -> dict:
    return dict(role="system", content=content)

def make_user_message(content: str) -> dict:
    return dict(role="user", content=content)

def make_assistant_message(content: str) -> dict:
    return dict(role="assistant", content=content)

class AbstractChatModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        pass
    
    def get_stats(self):
        return {}

@dataclass
class BaseModelArgs(ABC):
    """Base class for all model arguments"""
    model_name: str
    max_total_tokens: int = None
    max_input_tokens: int = None
    max_new_tokens: int = None
    temperature: float = 0.6
    top_p: float = None
    top_k: Optional[int] = None
    vision_support: bool = False
    log_probs: bool = False
    enable_thinking: bool = None
    reasoning_effort: Optional[str] = None

    @abstractmethod
    def make_model(self) -> AbstractChatModel:
        pass

    def prepare_server(self):
        pass

    def close_server(self):
        pass

@dataclass
class OpenAIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI model"""
    def make_model(self):
        return OpenAIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )

@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""
    deployment_name: str = None
    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            deployment_name=self.deployment_name,
            log_probs=self.log_probs
        )

@dataclass
class VLLMModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a local vLLM model."""
    port: int = 8000
    def make_model(self):
        return VLLMChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            port=self.port
        )

class ChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        top_p=None,
        top_k=None,
        max_retry=30,
        min_retry_wait_time=10,
        api_key_env_var=None,
        client_class=OpenAI,
        client_args=None,
        pricing_func=None,
        log_probs=False,
        enable_thinking=None,
        reasoning_effort=None,
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs
        self.enable_thinking = enable_thinking  # None/True/False
        self.reasoning_effort = reasoning_effort  # minimal/low/medium/high or None

        # Get the API key from the environment variable if not provided
        if api_key_env_var:
            api_key = api_key or os.getenv(api_key_env_var)
        self.api_key = api_key

        client_args = client_args or {}
        self.client = client_class(
            api_key=api_key,
            max_retries=10,
            **client_args,
        )
    
    def _is_reasoning_model(self) -> bool:
        """Whether the model uses max_completion_tokens."""
        n = self.model_name.lower()
        return (
            re.search(r"\bo[134]", n) is not None          # o1/o3/o4
            or re.search(r"gpt-5\.", n) is not None         # gpt-5.x
            or re.search(r"gpt-5-", n) is not None          # gpt-5-xxxx
        )

    def _reasoning_model_supports_temperature(self) -> bool:
        """Whether this reasoning model accepts a temperature setting."""
        n = self.model_name.lower()
        return (
            re.search(r"gpt-5\.", n) is not None
            or re.search(r"gpt-5-", n) is not None
        )

    def _should_enable_thinking(self) -> Optional[bool]:
        """Resolve the explicit thinking option or the model default."""
        n = self.model_name.lower()
        if self.enable_thinking is not None:
            return self.enable_thinking
        if "qwen3" in n:
            return True
        return None

    def _is_gemini_thinking(self) -> bool:
        """Whether the model uses Gemini thinking responses."""
        n = self.model_name.lower()
        return "gemini" in n and any(
            k in n for k in ["2.5-pro", "3-pro", "3.1-pro", "thinking"]
        )

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        e = None
        for itr in range(self.max_retry):
            self.retries += 1
            temperature = temperature if temperature is not None else self.temperature
            try:
                if self._is_reasoning_model():
                    create_kwargs = dict(
                        model=self.model_name,
                        messages=messages,
                        n=n_samples,
                        max_completion_tokens=self.max_tokens,
                        logprobs=self.log_probs,
                    )
                    if self._reasoning_model_supports_temperature():
                        create_kwargs["temperature"] = temperature
                        if self.top_p is not None:
                            create_kwargs["top_p"] = self.top_p
                elif self._should_enable_thinking() is not None:
                    thinking_val = self._should_enable_thinking()
                    extra = {
                        "enable_thinking": thinking_val,
                        "chat_template_kwargs": {"enable_thinking": thinking_val},
                    }
                    if "qwen3.6" in self.model_name.lower():
                        extra["chat_template_kwargs"]["preserve_thinking"] = True
                    create_kwargs = dict(
                        model=self.model_name,
                        messages=messages,
                        n=n_samples,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                        logprobs=self.log_probs,
                        extra_body=extra,
                    )
                    effective_top_p = self.top_p if self.top_p is not None else (0.95 if thinking_val else None)
                    if effective_top_p is not None:
                        create_kwargs["top_p"] = effective_top_p
                else:
                    create_kwargs = dict(
                        model=self.model_name,
                        messages=messages,
                        n=n_samples,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                        logprobs=self.log_probs,
                    )
                    if self.top_p is not None:
                        create_kwargs["top_p"] = self.top_p

                if self.reasoning_effort is not None:
                    create_kwargs["reasoning_effort"] = self.reasoning_effort
                model_lower = self.model_name.lower()
                if model_lower.startswith(("deepseek", "glm", "kimi", "gemini")):
                    create_kwargs.pop("extra_body", None)
                    if self.enable_thinking is not None:
                        if model_lower.startswith("gemini"):
                            if not self.enable_thinking:
                                create_kwargs["reasoning_effort"] = "none"
                        else:
                            thinking = {"type": "enabled" if self.enable_thinking else "disabled"}
                            if model_lower.startswith("glm") and self.enable_thinking:
                                thinking["clear_thinking"] = False
                            create_kwargs["extra_body"] = {"thinking": thinking}
                if self.top_k is not None:
                    create_kwargs.setdefault("extra_body", {})["top_k"] = self.top_k
                completion = self.client.chat.completions.create(**create_kwargs)
                if completion.choices and completion.choices[0].message is not None:
                    message = completion.choices[0].message
                    message.content = normalize_action_text(message.content)

                choice = completion.choices[0]
                if choice.message is None:
                    time.sleep(2)
                    continue
                res_content = choice.message.content

                has_execution_feedback = len(messages) > 2

                if not res_content or str(res_content).strip().lower() in ["", "null", "none"]:
                    time.sleep(2)
                    continue
                if "{" not in res_content and "<Final_answer>" not in res_content:
                    time.sleep(2)
                    continue
                if not has_execution_feedback and "<Final_answer>" in res_content and "{" not in res_content:
                    messages = messages + [{
                        "role": "user",
                        "content": (
                            "IMPORTANT: You must NOT submit a Final_answer without first executing code to verify "
                            "the result from the actual database. You have not run any code yet. "
                            "Please output a JSON execute_code action to query the data first."
                        )
                    }]
                    time.sleep(2)
                    continue
                if self.model_name.lower() == "gpt-5.4" and "{" in res_content:
                    try:
                        json_start = res_content.find("{")
                        first_json, json_end = json.JSONDecoder().raw_decode(res_content[json_start:])
                        prefix = res_content[:json_start].strip()
                        suffix = res_content[json_start + json_end:].strip()
                        is_execute_code = (
                            isinstance(first_json, dict)
                            and first_json.get("action") == "execute_code"
                            and isinstance(first_json.get("params"), dict)
                            and isinstance(first_json["params"].get("code"), str)
                        )
                        if is_execute_code and (prefix or suffix):
                            res_content = json.dumps(first_json, ensure_ascii=False)
                            completion.choices[0].message.content = res_content
                    except Exception:
                        pass

                if "{" in res_content and "<Final_answer>" in res_content and "</Final_answer>" in res_content:
                    if self.model_name.lower() == "gpt-5.4":
                        messages = messages + [{
                            "role": "user",
                            "content": (
                                "Your previous response mixed an execute_code JSON object with a Final_answer. "
                                "For GPT-5.4, this is invalid. Output exactly one JSON object and nothing else: "
                                "{\"action\": \"execute_code\", \"params\": {\"code\": \"<code>\"}}. "
                                "Do not include <Final_answer>, markdown, prose, or any second JSON object."
                            )
                        }]
                        time.sleep(2)
                        continue
                    fa_start = res_content.find("<Final_answer>")
                    res_content = res_content[fa_start:]
                    completion.choices[0].message.content = res_content

                self.success = True
                break
            except openai.OpenAIError as e:
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)
        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )
        input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0
        cost = {'input_tokens': input_tokens, 'completion_tokens': output_tokens}
        if n_samples == 1:
            res = completion.choices[0].message
            return res, cost
        else:
            return [c.message for c in completion.choices], cost
    
    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
        }

class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        top_p=None,
        top_k=None,
        max_tokens=100,
        max_retry=30,
        min_retry_wait_time=10,
        log_probs=False,
        enable_thinking=None,
        reasoning_effort=None,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENAI_API_KEY",
            client_class=OpenAI,
            log_probs=log_probs,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )

class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        deployment_name=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=30,
        min_retry_wait_time=10,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("API_VERSION")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"

        client_args = {
            "azure_deployment": deployment_name,
            "azure_endpoint": endpoint,
            "api_version": api_version,
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=AzureOpenAI,
            client_args=client_args,
            log_probs=log_probs,
        )

class VLLMChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        n_retry_server=10,
        min_retry_wait_time=60,
        port=8000,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=n_retry_server,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="VLLM_API_KEY",
            client_class=OpenAI,
            client_args={"base_url": "http://0.0.0.0:{}/v1".format(port)},
            pricing_func=None,
        )
