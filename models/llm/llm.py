import json
import logging
import math
from collections.abc import Generator, Sequence
from typing import Any, Optional, Union

import httpx

from dify_plugin import LargeLanguageModel
from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageRole,
    PromptMessageTool,
    ToolPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

from ..model_catalog import resolve_llm_model

logger = logging.getLogger(__name__)


class LlmProviderSampleLargeLanguageModel(LargeLanguageModel):
    """
    Azure OpenAI compatible large language model provider.
    """

    DEFAULT_NON_SSE_CHUNK = 4
    DEFAULT_SYNC_TIMEOUT = 30
    DEFAULT_ASYNC_TIMEOUT = 120
    DEFAULT_API_VERSION = "2024-02-15-preview"

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        """
        Invoke large language model.
        """
        config = self._resolve_llm_config(credentials)
        supports_sse = config.supports_sse
        effective_stream = stream and supports_sse

        try:
            api_base = self._get_api_base(credentials)
            deployment = self._get_deployment_name(model, credentials)
            params = self._build_query_params(credentials)
            headers = self._build_headers(credentials)
            messages = self._convert_prompt_messages(prompt_messages)
            payload = self._build_payload(
                model=model,
                messages=messages,
                model_parameters=model_parameters or {},
                tools=tools,
                stop=stop,
                stream=effective_stream,
                user=user,
            )
            timeout = self._select_timeout(credentials, effective_stream)
            url = self._build_chat_url(api_base, deployment)

            if effective_stream:
                return self._stream_chat(
                    url=url,
                    headers=headers,
                    params=params,
                    payload=payload,
                    timeout=timeout,
                    model=model,
                    prompt_messages=prompt_messages,
                )

            response_json = self._post_chat(
                url=url,
                headers=headers,
                params=params,
                payload=payload,
                timeout=timeout,
            )

            if stream and not supports_sse:
                return self._to_stream_from_completion(
                    response=response_json,
                    model=model,
                    prompt_messages=prompt_messages,
                    credentials=credentials,
                )

            return self._build_llm_result_from_response(
                response=response_json,
                model=model,
                prompt_messages=prompt_messages,
            )
        except CredentialsValidateFailedError:
            raise
        except httpx.HTTPStatusError as exc:
            logger.exception("HTTP error while invoking model %s", model)
            raise self._map_http_error(exc, is_validation=False) from exc
        except httpx.HTTPError as exc:
            logger.exception("HTTP error while invoking model %s", model)
            raise InvokeConnectionError(f"Provider request failed: {exc}") from exc
        except Exception:
            logger.exception("Unexpected error while invoking model %s", model)
            raise

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Estimate number of tokens based on prompt text length.
        """
        # Rough heuristic: assume 4 characters per token as a safe default.
        total_chars = sum(message.get_text_content().__len__() for message in prompt_messages)
        return max(0, math.ceil(total_chars / 4)) if total_chars > 0 else 0

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials by ensuring the deployment exists and credentials work.
        """
        try:
            deployment = self._get_deployment_name(model, credentials)
            self._check_deployment_exists(credentials=credentials, deployment=deployment)
        except CredentialsValidateFailedError:
            raise
        except httpx.HTTPStatusError as exc:
            logger.exception("HTTP error while validating credentials for model %s", model)
            raise self._map_http_error(exc, is_validation=True) from exc
        except httpx.HTTPError as exc:
            logger.exception("HTTP error while validating credentials for model %s", model)
            raise CredentialsValidateFailedError(str(exc)) from exc
        except Exception as ex:
            logger.exception("Unexpected error while validating credentials for model %s", model)
            raise CredentialsValidateFailedError(str(ex)) from ex

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _resolve_llm_config(self, credentials: dict):
        base_model_name = self._get_base_model_name(credentials)
        return resolve_llm_model(base_model_name)

    @staticmethod
    def _extract_optional_str(credentials: Any, key: str) -> Optional[str]:
        if not isinstance(credentials, dict):
            return None

        raw_value = credentials.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()

        nested_candidates = ("model", "model_credentials", "credentials")
        for nested_key in nested_candidates:
            nested = credentials.get(nested_key)
            if isinstance(nested, dict):
                nested_value = nested.get(key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
        return None

    def _get_api_base(self, credentials: dict) -> str:
        api_base = credentials.get("api_base")
        if not isinstance(api_base, str) or not api_base.strip():
            raise CredentialsValidateFailedError("API base is required.")
        return api_base.rstrip("/")

    def _get_deployment_name(self, model: str, credentials: dict) -> str:
        deployment_name = self._extract_optional_str(credentials, "deployment_name")
        if deployment_name:
            return deployment_name
        if not isinstance(model, str) or not model.strip():
            raise CredentialsValidateFailedError("Model name is required.")
        return model.strip()

    def _get_base_model_name(self, credentials: dict) -> str:
        base_model_name = self._extract_optional_str(credentials, "base_model_name")
        if not base_model_name:
            raise CredentialsValidateFailedError("Base model name is required.")
        return base_model_name

    def _build_query_params(self, credentials: dict) -> dict[str, Any]:
        api_version = credentials.get("api_version") or self.DEFAULT_API_VERSION
        return {"api-version": api_version}

    def _build_headers(self, credentials: dict) -> dict[str, str]:
        api_key = credentials.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise CredentialsValidateFailedError("API key is required.")
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key.strip(),
        }
        return headers

    def _build_chat_url(self, api_base: str, deployment: str) -> str:
        return f"{api_base}/openai/deployments/{deployment}/chat/completions"

    def _select_timeout(self, credentials: dict, stream: bool) -> float:
        sync_timeout = self._coerce_positive_int(credentials.get("sync_timeout"), self.DEFAULT_SYNC_TIMEOUT)
        async_timeout = self._coerce_positive_int(credentials.get("async_timeout"), self.DEFAULT_ASYNC_TIMEOUT)
        if async_timeout < sync_timeout:
            async_timeout = sync_timeout
        return float(async_timeout if stream else sync_timeout)

    @staticmethod
    def _coerce_positive_int(value: Any, default: int) -> int:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _convert_prompt_messages(self, prompt_messages: Sequence[PromptMessage]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in prompt_messages:
            role = message.role.value if isinstance(message.role, PromptMessageRole) else str(message.role)
            payload: dict[str, Any] = {"role": role.lower()}
            if message.name:
                payload["name"] = message.name
            payload["content"] = self._extract_text(message.content)
            if isinstance(message, ToolPromptMessage):
                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id:
                    payload["tool_call_id"] = tool_call_id
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                payload["tool_calls"] = self._convert_tool_calls(tool_calls)
            converted.append(payload)
        return converted

    def _convert_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool_call in tool_calls or []:
            if isinstance(tool_call, dict):
                converted.append(tool_call)
                continue
            function = getattr(tool_call, "function", None)
            converted.append(
                {
                    "id": getattr(tool_call, "id", "") or "",
                    "type": getattr(tool_call, "type", "function") or "function",
                    "function": {
                        "name": getattr(function, "name", "") if function else "",
                        "arguments": getattr(function, "arguments", "") if function else "",
                    },
                }
            )
        return converted

    def _convert_tools(self, tools: Optional[list[PromptMessageTool]]) -> Optional[list[dict[str, Any]]]:
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return converted

    def _build_payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        model_parameters: dict[str, Any],
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]],
        stream: bool,
        user: Optional[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "stream": stream,
        }

        parameters = {k: v for k, v in (model_parameters or {}).items() if v is not None}

        response_format = parameters.get("response_format")
        if isinstance(response_format, str):
            parameters["response_format"] = {"type": response_format}

        payload.update(parameters)

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        if stop:
            payload["stop"] = stop

        if user:
            payload["user"] = user

        return payload

    def _post_chat(
        self,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, params=params, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                self._handle_http_error(exc, is_validation=False)
            return response.json()

    def _stream_chat(
        self,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any],
        payload: dict[str, Any],
        timeout: float,
        model: str,
        prompt_messages: Sequence[PromptMessage],
    ) -> Generator[LLMResultChunk, None, None]:
        def iterator() -> Generator[LLMResultChunk, None, None]:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, headers=headers, params=params, json=payload) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        self._handle_http_error(exc, is_validation=False)
                    system_fingerprint: Optional[str] = None
                    for raw_line in response.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            logger.debug("Unable to decode SSE payload: %s", data)
                            continue
                        system_fingerprint = event.get("system_fingerprint", system_fingerprint)
                        for choice in event.get("choices", []):
                            yield self._build_chunk_from_event(
                                event=event,
                                choice=choice,
                                model=model,
                                prompt_messages=prompt_messages,
                                system_fingerprint=system_fingerprint,
                            )

        return iterator()

    def _build_chunk_from_event(
        self,
        event: dict[str, Any],
        choice: dict[str, Any],
        model: str,
        prompt_messages: Sequence[PromptMessage],
        system_fingerprint: Optional[str],
    ) -> LLMResultChunk:
        delta_dict = choice.get("delta", {})
        assistant_delta = self._build_assistant_delta(delta_dict)
        usage = None
        if choice.get("usage"):
            usage = self._build_usage(choice["usage"])
        return LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            system_fingerprint=system_fingerprint,
            delta=LLMResultChunkDelta(
                index=choice.get("index", 0),
                message=assistant_delta,
                usage=usage,
                finish_reason=choice.get("finish_reason"),
            ),
        )

    def _build_assistant_delta(self, delta: dict[str, Any]) -> AssistantPromptMessage:
        content = self._extract_text(delta.get("content"))
        tool_calls = self._convert_tool_calls(delta.get("tool_calls"))
        return AssistantPromptMessage(
            content=content,
            tool_calls=tool_calls,
        )

    def _build_llm_result_from_response(
        self,
        response: dict[str, Any],
        model: str,
        prompt_messages: Sequence[PromptMessage],
    ) -> LLMResult:
        choices = response.get("choices", [])
        if not choices:
            raise CredentialsValidateFailedError("Empty response received from model.")
        primary_choice = choices[0]
        assistant_message, reasoning_content = self._build_assistant_message_from_choice(primary_choice)
        usage = self._build_usage(response.get("usage"))
        return LLMResult(
            id=response.get("id"),
            model=model,
            prompt_messages=prompt_messages,
            message=assistant_message,
            usage=usage,
            system_fingerprint=response.get("system_fingerprint"),
            reasoning_content=reasoning_content,
        )

    def _build_assistant_message_from_choice(
        self, choice: dict[str, Any]
    ) -> tuple[AssistantPromptMessage, Optional[str]]:
        message = choice.get("message", {}) or {}
        content = self._extract_text(message.get("content"))
        tool_calls = self._convert_tool_calls(message.get("tool_calls"))
        reasoning_content = self._extract_text(message.get("reasoning_content"))
        return (
            AssistantPromptMessage(
                content=content,
                tool_calls=tool_calls,
            ),
            reasoning_content if reasoning_content else None,
        )

    def _build_usage(self, usage_dict: Any) -> LLMUsage:
        if not isinstance(usage_dict, dict):
            return LLMUsage.empty_usage()
        metadata = {
            "prompt_tokens": usage_dict.get("prompt_tokens", 0),
            "completion_tokens": usage_dict.get("completion_tokens", 0),
            "total_tokens": usage_dict.get("total_tokens", 0),
        }
        try:
            return LLMUsage.from_metadata(metadata)
        except AttributeError:
            # Older SDK versions may not expose from_metadata
            usage = LLMUsage.empty_usage()
            return usage.model_copy(
                update={
                    "prompt_tokens": metadata["prompt_tokens"],
                    "completion_tokens": metadata["completion_tokens"],
                    "total_tokens": metadata["total_tokens"],
                }
            )

    def _to_stream_from_completion(
        self,
        response: dict[str, Any],
        model: str,
        prompt_messages: Sequence[PromptMessage],
        credentials: dict,
    ) -> Generator[LLMResultChunk, None, None]:
        choices = response.get("choices", [])
        if not choices:
            raise CredentialsValidateFailedError("Empty response received from model.")
        primary_choice = choices[0]
        assistant_message, reasoning_content = self._build_assistant_message_from_choice(primary_choice)
        usage = self._build_usage(response.get("usage"))
        finish_reason = primary_choice.get("finish_reason")
        system_fingerprint = response.get("system_fingerprint")
        split_count = self._get_non_sse_chunk_count(credentials)
        segments = self._split_content(assistant_message.content, split_count)

        def iterator() -> Generator[LLMResultChunk, None, None]:
            for idx, segment in enumerate(segments):
                is_last = idx == len(segments) - 1
                message = AssistantPromptMessage(
                    content=segment,
                    tool_calls=assistant_message.tool_calls if is_last else [],
                )
                usage_payload = usage if is_last else None
                reasoning = reasoning_content if is_last else None
                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    system_fingerprint=system_fingerprint,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=message,
                        usage=usage_payload,
                        finish_reason=finish_reason if is_last else None,
                    ),
                )
                if reasoning and not segment:
                    # emit reasoning content if no textual segment in final chunk
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        system_fingerprint=system_fingerprint,
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=reasoning, tool_calls=[]),
                            usage=None,
                            finish_reason=finish_reason,
                        ),
                    )

        return iterator()

    def _get_non_sse_chunk_count(self, credentials: dict) -> int:
        return self._coerce_positive_int(credentials.get("non_sse_chunk_count"), self.DEFAULT_NON_SSE_CHUNK)

    @staticmethod
    def _split_content(text: str, parts: int) -> list[str]:
        if parts <= 1 or not text:
            return [text] if text else [""]
        step = max(1, math.ceil(len(text) / parts))
        segments = [text[i : i + step] for i in range(0, len(text), step)]
        return segments or [""]

    @staticmethod
    def _extract_text(content: Any) -> Any:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            multimodal_parts: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "reasoning"}:
                        text_parts.append(item.get("text", ""))
                    else:
                        multimodal_parts.append(item)
                elif hasattr(item, "data"):
                    item_type = getattr(item, "type", None)
                    if item_type == PromptMessageContentType.TEXT:
                        text_parts.append(getattr(item, "data", ""))
                    elif item_type == PromptMessageContentType.IMAGE:
                        data = getattr(item, "data", "")
                        if data:
                            multimodal_parts.append({"type": "image_url", "image_url": {"url": data}})
                    else:
                        text_parts.append(getattr(item, "data", ""))
            if multimodal_parts:
                if text_parts:
                    multimodal_parts.insert(0, {"type": "text", "text": "".join(text_parts)})
                return multimodal_parts
            return "".join(text_parts)
        return str(content)

    def _handle_http_error(self, exc: httpx.HTTPStatusError, is_validation: bool) -> None:
        raise self._map_http_error(exc, is_validation)

    def _map_http_error(self, exc: httpx.HTTPStatusError, is_validation: bool) -> Exception:
        status = exc.response.status_code
        message = self._extract_error_message(exc.response)
        logger.error("Provider request failed with status %s: %s", status, message)

        # Treat authentication/authorization failures as credential issues when validating.
        if status in {401, 403}:
            if is_validation:
                return CredentialsValidateFailedError(f"Authentication failed ({status}): {message}")
            return InvokeAuthorizationError(f"Authentication failed ({status}): {message}")

        # During validation, propagate other HTTP errors as credential validation issues.
        if is_validation:
            return CredentialsValidateFailedError(f"Provider request failed ({status}): {message}")

        if status == 429:
            return InvokeRateLimitError(f"Provider request failed ({status}): {message}")
        if status >= 500:
            return InvokeServerUnavailableError(f"Provider request failed ({status}): {message}")
        if status in {400, 404, 422}:
            return InvokeBadRequestError(f"Provider request failed ({status}): {message}")

        return InvokeError(f"Provider request failed ({status}): {message}")

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeAuthorizationError: [InvokeAuthorizationError],
            InvokeRateLimitError: [InvokeRateLimitError],
            InvokeServerUnavailableError: [InvokeServerUnavailableError, httpx.TimeoutException, httpx.ConnectError],
            InvokeBadRequestError: [InvokeBadRequestError],
            InvokeConnectionError: [InvokeConnectionError, httpx.TransportError],
            ValueError: [ValueError],
            InvokeError: [InvokeError, Exception],
        }

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                if "error" in payload and isinstance(payload["error"], dict):
                    return payload["error"].get("message") or json.dumps(payload["error"])
                return json.dumps(payload)
        except Exception:
            pass
        return response.text

    def _check_deployment_exists(self, credentials: dict, deployment: str) -> None:
        api_base = self._get_api_base(credentials)
        headers = self._build_headers(credentials)
        params = self._build_query_params(credentials)
        url = f"{api_base}/openai/deployments"
        timeout = self._select_timeout(credentials, stream=False)
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers, params=params)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                self._handle_http_error(exc, is_validation=True)
            try:
                data = response.json()
            except ValueError as exc:
                raise CredentialsValidateFailedError("Invalid JSON response while validating deployment.") from exc

        deployments = data.get("data") if isinstance(data, dict) else None
        if deployments is None and isinstance(data, dict):
            deployments = data.get("value")

        if isinstance(deployments, list) and deployments:
            names = {
                item.get("id") or item.get("name")
                for item in deployments
                if isinstance(item, dict)
            }
            if names and deployment not in names:
                raise CredentialsValidateFailedError(
                    f"Deployment '{deployment}' not found. Available deployments: {', '.join(sorted(names))}"
                )

    def get_customizable_model_schema(self, model: str, credentials: dict) -> Optional[AIModelEntity]:
        config = self._resolve_llm_config(credentials)
        return config.clone_with_deployment(model)
