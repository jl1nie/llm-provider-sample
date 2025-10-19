from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict

from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    PARAMETER_RULE_TEMPLATE,
    PriceConfig,
)
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.entities.model.llm import LLMMode


def _temperature_rule() -> ParameterRule:
    return ParameterRule(
        name="temperature",
        **PARAMETER_RULE_TEMPLATE[DefaultParameterName.TEMPERATURE],
    )


def _top_p_rule() -> ParameterRule:
    return ParameterRule(
        name="top_p",
        **PARAMETER_RULE_TEMPLATE[DefaultParameterName.TOP_P],
    )


def _presence_penalty_rule() -> ParameterRule:
    return ParameterRule(
        name="presence_penalty",
        **PARAMETER_RULE_TEMPLATE[DefaultParameterName.PRESENCE_PENALTY],
    )


def _frequency_penalty_rule() -> ParameterRule:
    return ParameterRule(
        name="frequency_penalty",
        **PARAMETER_RULE_TEMPLATE[DefaultParameterName.FREQUENCY_PENALTY],
    )


def _max_tokens_rule(default: int, minimum: int, maximum: int, *, field_name: str = "max_tokens") -> ParameterRule:
    rule = ParameterRule(
        name=field_name,
        **PARAMETER_RULE_TEMPLATE[DefaultParameterName.MAX_TOKENS],
    )
    rule.default = default
    rule.min = minimum
    rule.max = maximum
    return rule


def _response_format_rule() -> ParameterRule:
    return ParameterRule(
        name="response_format",
        use_template="response_format",
    )


@dataclass(frozen=True)
class LlmModelConfig:
    base_model_name: str
    entity: AIModelEntity
    supports_sse: bool

    def clone_with_deployment(self, deployment: str) -> AIModelEntity:
        entity_copy = deepcopy(self.entity)
        entity_copy.model = deployment
        if entity_copy.label is None:
            entity_copy.label = I18nObject(en_US=deployment)
        else:
            for attr in ("en_US", "zh_Hans", "ja_JP"):
                if hasattr(entity_copy.label, attr):
                    setattr(entity_copy.label, attr, deployment)
        return entity_copy


LLM_MODELS: Dict[str, LlmModelConfig] = {
    "openai4.1": LlmModelConfig(
        base_model_name="openai4.1",
        supports_sse=True,
        entity=AIModelEntity(
            model="openai4.1",
            label=I18nObject(en_US="openai4.1", zh_Hans="openai4.1", ja_JP="openai4.1"),
            model_type=ModelType.LLM,
            features=[
                ModelFeature.MULTI_TOOL_CALL,
                ModelFeature.AGENT_THOUGHT,
                ModelFeature.STREAM_TOOL_CALL,
            ],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
                ModelPropertyKey.CONTEXT_SIZE: 128000,
            },
            parameter_rules=[
                _temperature_rule(),
                _top_p_rule(),
                _presence_penalty_rule(),
                _frequency_penalty_rule(),
                _max_tokens_rule(default=4096, minimum=1, maximum=128000),
                _response_format_rule(),
            ],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=Decimal("0"),
                unit=Decimal("0.001"),
                currency="USD",
            ),
        ),
    ),
    "openai5": LlmModelConfig(
        base_model_name="openai5",
        supports_sse=True,
        entity=AIModelEntity(
            model="openai5",
            label=I18nObject(en_US="openai5", zh_Hans="openai5", ja_JP="openai5"),
            model_type=ModelType.LLM,
            features=[
                ModelFeature.MULTI_TOOL_CALL,
                ModelFeature.AGENT_THOUGHT,
                ModelFeature.STREAM_TOOL_CALL,
            ],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
                ModelPropertyKey.CONTEXT_SIZE: 200000,
            },
            parameter_rules=[
                _temperature_rule(),
                _top_p_rule(),
                _presence_penalty_rule(),
                _frequency_penalty_rule(),
                _max_tokens_rule(default=8192, minimum=1, maximum=200000),
                _response_format_rule(),
            ],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=Decimal("0"),
                unit=Decimal("0.001"),
                currency="USD",
            ),
        ),
    ),
    "claude-v4sonnet": LlmModelConfig(
        base_model_name="claude-v4sonnet",
        supports_sse=False,
        entity=AIModelEntity(
            model="claude-v4sonnet",
            label=I18nObject(en_US="claude-v4sonnet", zh_Hans="claude-v4sonnet", ja_JP="claude-v4sonnet"),
            model_type=ModelType.LLM,
            features=[
                ModelFeature.MULTI_TOOL_CALL,
            ],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
                ModelPropertyKey.CONTEXT_SIZE: 200000,
            },
            parameter_rules=[
                _temperature_rule(),
                _top_p_rule(),
                _presence_penalty_rule(),
                _frequency_penalty_rule(),
                _max_tokens_rule(default=4096, minimum=1, maximum=4096),
                _response_format_rule(),
            ],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=Decimal("0"),
                unit=Decimal("0.001"),
                currency="USD",
            ),
        ),
    ),
    "gemini-2.5-flash": LlmModelConfig(
        base_model_name="gemini-2.5-flash",
        supports_sse=False,
        entity=AIModelEntity(
            model="gemini-2.5-flash",
            label=I18nObject(en_US="gemini-2.5-flash", zh_Hans="gemini-2.5-flash", ja_JP="gemini-2.5-flash"),
            model_type=ModelType.LLM,
            features=[
                ModelFeature.MULTI_TOOL_CALL,
            ],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
                ModelPropertyKey.CONTEXT_SIZE: 1048576,
            },
            parameter_rules=[
                _temperature_rule(),
                _top_p_rule(),
                _presence_penalty_rule(),
                _frequency_penalty_rule(),
                _max_tokens_rule(default=8192, minimum=1, maximum=1048576),
                _response_format_rule(),
            ],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=Decimal("0"),
                unit=Decimal("0.001"),
                currency="USD",
            ),
        ),
    ),
    "gemini-2.5-pro": LlmModelConfig(
        base_model_name="gemini-2.5-pro",
        supports_sse=True,
        entity=AIModelEntity(
            model="gemini-2.5-pro",
            label=I18nObject(en_US="gemini-2.5-pro", zh_Hans="gemini-2.5-pro", ja_JP="gemini-2.5-pro"),
            model_type=ModelType.LLM,
            features=[
                ModelFeature.MULTI_TOOL_CALL,
                ModelFeature.AGENT_THOUGHT,
                ModelFeature.STREAM_TOOL_CALL,
            ],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
                ModelPropertyKey.CONTEXT_SIZE: 2097152,
            },
            parameter_rules=[
                _temperature_rule(),
                _top_p_rule(),
                _presence_penalty_rule(),
                _frequency_penalty_rule(),
                _max_tokens_rule(default=32768, minimum=1, maximum=2097152),
                _response_format_rule(),
            ],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=Decimal("0"),
                unit=Decimal("0.001"),
                currency="USD",
            ),
        ),
    ),
}


def resolve_llm_model(base_model_name: str) -> LlmModelConfig:
    config = LLM_MODELS.get(base_model_name)
    if not config:
        raise CredentialsValidateFailedError(f"Unsupported base model '{base_model_name}'.")
    return config


@dataclass(frozen=True)
class EmbeddingModelConfig:
    base_model_name: str
    entity: AIModelEntity

    def clone_with_deployment(self, deployment: str) -> AIModelEntity:
        entity_copy = deepcopy(self.entity)
        entity_copy.model = deployment
        if entity_copy.label is None:
            entity_copy.label = I18nObject(en_US=deployment)
        else:
            for attr in ("en_US", "zh_Hans", "ja_JP"):
                if hasattr(entity_copy.label, attr):
                    setattr(entity_copy.label, attr, deployment)
        return entity_copy


EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
    "text-embedding-3-small": EmbeddingModelConfig(
        base_model_name="text-embedding-3-small",
        entity=AIModelEntity(
            model="text-embedding-3-small",
            label=I18nObject(
                en_US="text-embedding-3-small",
                zh_Hans="text-embedding-3-small",
                ja_JP="text-embedding-3-small",
            ),
            model_type=ModelType.TEXT_EMBEDDING,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8191,
                ModelPropertyKey.MAX_CHUNKS: 32,
            },
            parameter_rules=[],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=None,
                unit=Decimal("0.001"),
                currency="USD",
            ),
        ),
    ),
    "text-embedding-3-large": EmbeddingModelConfig(
        base_model_name="text-embedding-3-large",
        entity=AIModelEntity(
            model="text-embedding-3-large",
            label=I18nObject(
                en_US="text-embedding-3-large",
                zh_Hans="text-embedding-3-large",
                ja_JP="text-embedding-3-large",
            ),
            model_type=ModelType.TEXT_EMBEDDING,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8191,
                ModelPropertyKey.MAX_CHUNKS: 32,
            },
            parameter_rules=[],
            pricing=PriceConfig(
                input=Decimal("0"),
                output=None,
                unit=Decimal("0.001"),
                currency="USD",
            ),
        ),
    ),
}


def resolve_embedding_model(base_model_name: str) -> EmbeddingModelConfig:
    config = EMBEDDING_MODELS.get(base_model_name)
    if not config:
        raise CredentialsValidateFailedError(f"Unsupported embedding base model '{base_model_name}'.")
    return config
