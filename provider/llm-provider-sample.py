import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class LlmProviderSampleModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping[str, Any]) -> None:
        """
        Validate provider credentials. Raises CredentialsValidateFailedError if invalid.
        """
        try:
            api_base = self._extract_required_str(
                credentials, ("openai_api_base", "api_base")
            )
            self._validate_url(api_base)
            api_key = self._extract_required_str(
                credentials, ("openai_api_key", "api_key")
            )
            if not api_key:
                raise CredentialsValidateFailedError("API key is required.")

            sync_timeout = self._coerce_positive_float(
                credentials.get("sync_timeout"), default=30.0, field="sync_timeout"
            )
            async_timeout = self._coerce_positive_float(
                credentials.get("async_timeout"), default=120.0, field="async_timeout"
            )
            if async_timeout < sync_timeout:
                raise CredentialsValidateFailedError(
                    "Stream timeout must be greater than or equal to sync timeout."
                )
            self._coerce_positive_int(
                credentials.get("non_sse_chunk_count"),
                default=4,
                field="non_sse_chunk_count",
            )
        except CredentialsValidateFailedError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "%s credentials validate failed", self.get_provider_schema().provider
            )
            raise exc

    @staticmethod
    def _extract_required_str(
        credentials: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> str:
        for key in keys:
            value = credentials.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise CredentialsValidateFailedError(f"{keys[0]} is required.")

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CredentialsValidateFailedError(
                "API base must be a valid HTTP(S) URL."
            )

    @staticmethod
    def _coerce_positive_int(value: Any, *, default: int, field: str) -> int:
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            raise CredentialsValidateFailedError(
                f"{field} must be a positive integer."
            )
        if isinstance(value, int):
            candidate = value
        elif isinstance(value, str):
            candidate = int(value.strip()) if value.strip().isdigit() else None
        else:
            candidate = None
        if candidate is None or candidate <= 0:
            raise CredentialsValidateFailedError(
                f"{field} must be a positive integer."
            )
        return candidate

    @staticmethod
    def _coerce_positive_float(value: Any, *, default: float, field: str) -> float:
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            raise CredentialsValidateFailedError(
                f"{field} must be a positive number."
            )
        if isinstance(value, (int, float)):
            candidate = float(value)
        elif isinstance(value, str):
            try:
                candidate = float(value.strip())
            except ValueError as exc:  # pragma: no cover - defensive
                raise CredentialsValidateFailedError(
                    f"{field} must be a positive number."
                ) from exc
        else:
            raise CredentialsValidateFailedError(
                f"{field} must be a positive number."
            )
        if candidate <= 0:
            raise CredentialsValidateFailedError(
                f"{field} must be a positive number."
            )
        return candidate
