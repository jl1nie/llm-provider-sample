import openai
from httpx import Timeout

from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from .constants import AZURE_OPENAI_API_VERSION


class _CommonAzureOpenAI:
    @staticmethod
    def _to_credential_kwargs(credentials: dict) -> dict:
        api_version = credentials.get(
            "openai_api_version",
            credentials.get("api_version", AZURE_OPENAI_API_VERSION),
        )
        api_key = credentials.get("openai_api_key") or credentials.get("api_key")
        api_base = credentials.get("openai_api_base") or credentials.get("api_base")
        if not api_key:
            raise ValueError("Azure OpenAI API key is required.")
        if not api_base:
            raise ValueError("Azure OpenAI API base endpoint is required.")

        sync_timeout = _CommonAzureOpenAI._get_timeout_seconds(
            credentials, "sync_timeout", default=30.0
        )
        async_timeout = _CommonAzureOpenAI._get_timeout_seconds(
            credentials, "async_timeout", default=120.0
        )
        max_timeout = max(sync_timeout, async_timeout)
        credentials_kwargs = {
            "api_key": api_key,
            "azure_endpoint": api_base,
            "api_version": api_version,
            "timeout": Timeout(max_timeout, read=async_timeout, write=10.0, connect=5.0),
            "max_retries": 1,
        }

        return credentials_kwargs

    @staticmethod
    def _get_timeout_seconds(credentials: dict, field: str, default: float) -> float:
        raw_value = credentials.get(field)
        if raw_value in (None, ""):
            return float(default)
        if isinstance(raw_value, bool):
            raise ValueError(f"{field} must be a positive number.")
        if isinstance(raw_value, (int, float)):
            value = float(raw_value)
        elif isinstance(raw_value, str):
            try:
                value = float(raw_value.strip())
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{field} must be a positive number.") from exc
        else:
            raise ValueError(f"{field} must be a positive number.")
        if value <= 0:
            raise ValueError(f"{field} must be a positive number.")
        return value

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [openai.APIConnectionError, openai.APITimeoutError],
            InvokeServerUnavailableError: [openai.InternalServerError],
            InvokeRateLimitError: [openai.RateLimitError],
            InvokeAuthorizationError: [
                openai.AuthenticationError,
                openai.PermissionDeniedError,
            ],
            InvokeBadRequestError: [
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
                openai.APIError,
            ],
        }
