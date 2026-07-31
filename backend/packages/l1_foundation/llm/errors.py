from __future__ import annotations


class LlmError(RuntimeError):
    """Base error raised by an L1 language-model provider."""


class LlmConfigurationError(LlmError):
    """The selected provider is missing required configuration."""


class LlmResponseError(LlmError):
    """The provider returned an invalid or unsuccessful response."""


class UnsupportedResponseFormatError(LlmError):
    """The selected provider cannot honor the requested response format."""

