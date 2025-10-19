import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class LlmProviderSampleModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            api_base = credentials.get("api_base")
            api_key = credentials.get("api_key")
            if not isinstance(api_base, str) or not api_base.strip():
                raise CredentialsValidateFailedError("API base endpoint is required.")

            parsed = urlparse(api_base.strip())
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise CredentialsValidateFailedError("API base must be a valid HTTP(S) URL.")

            if not isinstance(api_key, str) or not api_key.strip():
                raise CredentialsValidateFailedError("API key is required.")

            for field in ("sync_timeout", "async_timeout", "non_sse_chunk_count"):
                self._validate_positive_int(credentials, field)

            if (
                isinstance(credentials.get("async_timeout"), int)
                and isinstance(credentials.get("sync_timeout"), int)
                and credentials["async_timeout"] < credentials["sync_timeout"]
            ):
                raise CredentialsValidateFailedError("Stream timeout must be greater than or equal to sync timeout.")
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise ex

    @staticmethod
    def _validate_positive_int(credentials: Mapping[str, Any], field: str) -> None:
        """
        Validate a credential field that should be a positive integer when provided.
        """
        if field not in credentials:
            return
        value = credentials.get(field)
        if value is None or value == "":
            return
        if isinstance(value, bool):
            raise CredentialsValidateFailedError(f"{field} must be a positive integer.")
        if isinstance(value, str):
            if not value.strip():
                return
            if not value.isdigit():
                raise CredentialsValidateFailedError(f"{field} must be a positive integer.")
            value = int(value)
        if not isinstance(value, int):
            raise CredentialsValidateFailedError(f"{field} must be a positive integer.")
        if value <= 0:
            raise CredentialsValidateFailedError(f"{field} must be a positive integer.")
