"""Recording-summary pipeline stage and its manual regeneration use case."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from l2_core.audio_processing.stages.summary.regeneration import (
        RecordingSummaryNotReadyError,
        RecordingSummaryRegenerationService,
    )
    from l2_core.audio_processing.stages.summary.stage import GenerateSummaryStage

__all__ = ["GenerateSummaryStage", "RecordingSummaryNotReadyError", "RecordingSummaryRegenerationService"]


def __getattr__(name: str) -> Any:
    if name == "GenerateSummaryStage":
        from l2_core.audio_processing.stages.summary.stage import GenerateSummaryStage

        return GenerateSummaryStage
    if name in {"RecordingSummaryNotReadyError", "RecordingSummaryRegenerationService"}:
        from l2_core.audio_processing.stages.summary.regeneration import (
            RecordingSummaryNotReadyError,
            RecordingSummaryRegenerationService,
        )

        return {
            "RecordingSummaryNotReadyError": RecordingSummaryNotReadyError,
            "RecordingSummaryRegenerationService": RecordingSummaryRegenerationService,
        }[name]
    raise AttributeError(name)
