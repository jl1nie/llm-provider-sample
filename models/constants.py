from __future__ import annotations

from pydantic import BaseModel

from dify_plugin.entities.model import (
    PARAMETER_RULE_TEMPLATE,
    AIModelEntity,
    DefaultParameterName,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    PriceConfig,
)
from dify_plugin.entities.model.llm import LLMMode

AZURE_OPENAI_API_VERSION = "2024-02-15-preview"


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


def _max_tokens_rule(default: int, minimum: int, maximum: int) -> ParameterRule:
    rule = ParameterRule(
        name="max_tokens",
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


class AzureBaseModel(BaseModel):
    base_model_name: str
    entity: AIModelEntity
    supports_streaming: bool = True


LLM_BASE_MODELS = [
    AzureBaseModel(
        base_model_name="claude-v4sonnet",
        supports_streaming=False,
        entity=AIModelEntity(
            model="fake-deployment-name",
            label=I18nObject(
                en_US="claude-v4sonnet",
                zh_Hans="claude-v4sonnet",
                ja_JP="claude-v4sonnet",
            ),
            model_type=ModelType.LLM,
            features=[
                ModelFeature.AGENT_THOUGHT,
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
                input=0,
                output=0,
                unit=0.001,
                currency="USD",
            ),
        ),
    ),
    AzureBaseModel(
        base_model_name="openai4.1",
        supports_streaming=True,
        entity=AIModelEntity(
            model="fake-deployment-name",
            label=I18nObject(
                en_US="openai4.1",
                zh_Hans="openai4.1",
                ja_JP="openai4.1",
            ),
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
                input=0,
                output=0,
                unit=0.001,
                currency="USD",
            ),
        ),
    ),
    AzureBaseModel(
        base_model_name="openai5",
        supports_streaming=True,
        entity=AIModelEntity(
            model="fake-deployment-name",
            label=I18nObject(
                en_US="openai5",
                zh_Hans="openai5",
                ja_JP="openai5",
            ),
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
                input=0,
                output=0,
                unit=0.001,
                currency="USD",
            ),
        ),
    ),
    AzureBaseModel(
        base_model_name="gemini-2.5-flash",
        supports_streaming=False,
        entity=AIModelEntity(
            model="fake-deployment-name",
            label=I18nObject(
                en_US="gemini-2.5-flash",
                zh_Hans="gemini-2.5-flash",
                ja_JP="gemini-2.5-flash",
            ),
            model_type=ModelType.LLM,
            features=[
                ModelFeature.AGENT_THOUGHT,
            ],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
                ModelPropertyKey.CONTEXT_SIZE: 100000,
            },
            parameter_rules=[
                _temperature_rule(),
                _top_p_rule(),
                _presence_penalty_rule(),
                _frequency_penalty_rule(),
                _max_tokens_rule(default=4096, minimum=1, maximum=100000),
                _response_format_rule(),
            ],
            pricing=PriceConfig(
                input=0,
                output=0,
                unit=0.001,
                currency="USD",
            ),
        ),
    ),
    AzureBaseModel(
        base_model_name="gemini-2.5-pro",
        supports_streaming=True,
        entity=AIModelEntity(
            model="fake-deployment-name",
            label=I18nObject(
                en_US="gemini-2.5-pro",
                zh_Hans="gemini-2.5-pro",
                ja_JP="gemini-2.5-pro",
            ),
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
                input=0,
                output=0,
                unit=0.001,
                currency="USD",
            ),
        ),
    ),
]


EMBEDDING_BASE_MODELS = [
    AzureBaseModel(
        base_model_name="text-embedding-3-small",
        entity=AIModelEntity(
            model="fake-deployment-name",
            label=I18nObject(
                en_US="text-embedding-3-small",
                zh_Hans="text-embedding-3-small",
                ja_JP="text-embedding-3-small",
            ),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.TEXT_EMBEDDING,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8191,
                ModelPropertyKey.MAX_CHUNKS: 32,
            },
            pricing=PriceConfig(
                input=0,
                unit=0.001,
                currency="USD",
            ),
        ),
    ),
    AzureBaseModel(
        base_model_name="text-embedding-3-large",
        entity=AIModelEntity(
            model="fake-deployment-name",
            label=I18nObject(
                en_US="text-embedding-3-large",
                zh_Hans="text-embedding-3-large",
                ja_JP="text-embedding-3-large",
            ),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.TEXT_EMBEDDING,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8191,
                ModelPropertyKey.MAX_CHUNKS: 32,
            },
            pricing=PriceConfig(
                input=0,
                unit=0.001,
                currency="USD",
            ),
        ),
    ),
]


def get_llm_base_model(base_model_name: str) -> AzureBaseModel | None:
    for candidate in LLM_BASE_MODELS:
        if candidate.base_model_name == base_model_name:
            return candidate
    return None


def get_embedding_base_model(base_model_name: str) -> AzureBaseModel | None:
    for candidate in EMBEDDING_BASE_MODELS:
        if candidate.base_model_name == base_model_name:
            return candidate
    return None
