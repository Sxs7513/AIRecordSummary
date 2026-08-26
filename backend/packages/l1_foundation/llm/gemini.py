from __future__ import annotations

from collections.abc import Sequence

from l1_foundation.llm.contracts import ChatMessage, CompletionOptions, JsonObject, JsonValue, LlmProvider
from l1_foundation.llm.openai_compatible import OpenAiCompatibleLanguageModel, SynchronousRequestRateLimiter

_SUPPORTED_SCHEMA_SCALAR_KEYS = (
    "type",
    "title",
    "description",
    "enum",
    "format",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
)


class GeminiLanguageModel(OpenAiCompatibleLanguageModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds: float = 300.0,
        min_request_interval_seconds: float = 5.0,
        request_rate_limiter: SynchronousRequestRateLimiter | None = None,
    ) -> None:
        super().__init__(
            provider=LlmProvider.GEMINI,
            provider_label="Gemini",
            api_key_name="GEMINI_API_KEY",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            strict_json_schema=True,
            max_temperature=2,
            include_temperature=False,
            stream_include_usage=True,
            rate_limit_max_attempts=3,
            rate_limit_retry_seconds=10,
            min_request_interval_seconds=min_request_interval_seconds,
            request_rate_limiter=request_rate_limiter,
        )

    def _payload(self, messages: Sequence[ChatMessage], options: CompletionOptions, *, stream: bool) -> JsonObject:
        payload = super()._payload(messages, options, stream=stream)
        request_model = str(payload["model"]).lower()
        payload["reasoning_effort"] = "low" if request_model.startswith("gemini-3.7-flash") else "minimal"
        return payload

    def _json_schema_for_provider(self, schema: JsonObject) -> JsonObject:
        """Return Gemini's supported strict JSON Schema subset."""

        raw_defs = schema.get("$defs")
        definitions = raw_defs if isinstance(raw_defs, dict) else {}

        def normalize(raw: JsonValue) -> JsonObject:
            if not isinstance(raw, dict):
                return {}
            node = raw
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                target = definitions.get(reference.removeprefix("#/$defs/"))
                return normalize(target)

            normalized: JsonObject = {
                key: node[key] for key in _SUPPORTED_SCHEMA_SCALAR_KEYS if key in node
            }
            raw_any_of = node.get("anyOf")
            if isinstance(raw_any_of, list):
                any_of_items = raw_any_of

                def is_null_schema(item: JsonValue) -> bool:
                    if not isinstance(item, dict):
                        return False
                    return item.get("type") == "null"

                null_count = sum(1 for item in any_of_items if is_null_schema(item))
                value_items = [item for item in any_of_items if not is_null_schema(item)]
                if null_count == 1 and len(value_items) == 1:
                    nullable = normalize(value_items[0])
                    value_type = nullable.get("type")
                    if isinstance(value_type, str):
                        nullable["type"] = [value_type, "null"]
                        nullable_enum = nullable.get("enum")
                        if isinstance(nullable_enum, list) and None not in nullable_enum:
                            nullable["enum"] = [*nullable_enum, None]
                        nullable.update({key: value for key, value in normalized.items() if key != "type"})
                        normalized = nullable
                    else:
                        normalized["anyOf"] = [normalize(item) for item in any_of_items]
                else:
                    normalized["anyOf"] = [normalize(item) for item in any_of_items]
            raw_properties = node.get("properties")
            if isinstance(raw_properties, dict):
                properties: JsonObject = {key: normalize(value) for key, value in raw_properties.items()}
                normalized["properties"] = properties
                raw_required: object = node.get("required")
                if isinstance(raw_required, list):
                    normalized["required"] = [
                        name
                        for name in raw_required
                        if isinstance(name, str) and name in properties
                    ]
                normalized["additionalProperties"] = False
            raw_items = node.get("items")
            if isinstance(raw_items, dict):
                normalized["items"] = normalize(raw_items)
            raw_prefix_items = node.get("prefixItems")
            if isinstance(raw_prefix_items, list):
                normalized["prefixItems"] = [normalize(item) for item in raw_prefix_items]
            return normalized

        return normalize(schema)
