import logging
import math
import json
from decimal import Decimal
from typing import Any, Optional

import httpx

from dify_plugin import TextEmbeddingModel
from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.text_embedding import EmbeddingUsage, TextEmbeddingResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

from ..model_catalog import resolve_embedding_model

logger = logging.getLogger(__name__)


class LlmProviderSampleTextEmbeddingModel(TextEmbeddingModel):
    """
    Azure OpenAI compatible text embedding model provider.
    """

    DEFAULT_SYNC_TIMEOUT = 30
    DEFAULT_API_VERSION = "2024-02-15-preview"

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: Optional[str] = None,
    ) -> TextEmbeddingResult:
        if not texts:
            raise CredentialsValidateFailedError("At least one text input is required.")

        self._resolve_embedding_config(credentials)
        api_base = self._get_api_base(credentials)
        deployment = self._get_deployment_name(model, credentials)
        params = self._build_query_params(credentials)
        headers = self._build_headers(credentials)
        payload = {
            "input": texts,
            "model": model,
        }
        if user:
            payload["user"] = user
        if input_type:
            payload["input_type"] = input_type

        timeout = self._select_timeout(credentials)
        url = f"{api_base}/openai/deployments/{deployment}/embeddings"

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, params=params, json=payload)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    self._handle_http_error(exc, is_validation=False)
                data = response.json()
        except httpx.HTTPError as exc:
            logger.exception("HTTP error while invoking embedding model %s", model)
            raise InvokeConnectionError(f"Embedding request failed: {exc}") from exc

        embeddings = [
            item.get("embedding", [])
            for item in data.get("data", [])
            if isinstance(item, dict)
        ]

        usage = self._build_usage(data.get("usage"))

        return TextEmbeddingResult(
            model=model,
            embeddings=embeddings,
            usage=usage,
        )

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> list[int]:
        """
        Estimate token counts for embedding inputs.
        """
        return [max(0, math.ceil(len(text) / 4)) if text else 0 for text in texts]

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate embedding credentials by confirming deployment access.
        """
        try:
            self._resolve_embedding_config(credentials)
            deployment = self._get_deployment_name(model, credentials)
            self._check_deployment_exists(credentials, deployment)
        except CredentialsValidateFailedError:
            raise
        except httpx.HTTPError as exc:
            logger.exception("HTTP error while validating embedding credentials for model %s", model)
            raise CredentialsValidateFailedError(str(exc)) from exc
        except Exception as ex:
            logger.exception("Unexpected error while validating embedding credentials for model %s", model)
            raise CredentialsValidateFailedError(str(ex)) from ex

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_embedding_config(self, credentials: dict):
        base_model_name = self._get_base_model_name(credentials)
        return resolve_embedding_model(base_model_name)

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
        return {
            "Content-Type": "application/json",
            "api-key": api_key.strip(),
        }

    def _select_timeout(self, credentials: dict) -> float:
        return float(self._coerce_positive_int(credentials.get("sync_timeout"), self.DEFAULT_SYNC_TIMEOUT))

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

    def _build_usage(self, usage_dict: Any) -> EmbeddingUsage:
        tokens = 0
        total_tokens = 0
        if isinstance(usage_dict, dict):
            tokens = usage_dict.get("prompt_tokens", 0)
            total_tokens = usage_dict.get("total_tokens", tokens)
        return EmbeddingUsage(
            tokens=tokens,
            total_tokens=total_tokens,
            unit_price=Decimal("0"),
            price_unit=Decimal("0"),
            total_price=Decimal("0"),
            currency="USD",
            latency=0.0,
        )

    def _handle_http_error(self, exc: httpx.HTTPStatusError, is_validation: bool) -> None:
        raise self._map_http_error(exc, is_validation)

    def _map_http_error(self, exc: httpx.HTTPStatusError, is_validation: bool) -> Exception:
        status = exc.response.status_code
        message = self._extract_error_message(exc.response)
        logger.error("Embedding request failed with status %s: %s", status, message)

        if status in {401, 403}:
            if is_validation:
                return CredentialsValidateFailedError(f"Authentication failed ({status}): {message}")
            return InvokeAuthorizationError(f"Authentication failed ({status}): {message}")

        if is_validation:
            return CredentialsValidateFailedError(f"Embedding request failed ({status}): {message}")

        if status == 429:
            return InvokeRateLimitError(f"Embedding request failed ({status}): {message}")
        if status >= 500:
            return InvokeServerUnavailableError(f"Embedding request failed ({status}): {message}")
        if status in {400, 404, 422}:
            return InvokeBadRequestError(f"Embedding request failed ({status}): {message}")

        return InvokeError(f"Embedding request failed ({status}): {message}")

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
        timeout = self._select_timeout(credentials)

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
        config = self._resolve_embedding_config(credentials)
        return config.clone_with_deployment(model)
