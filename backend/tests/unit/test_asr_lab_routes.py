from uuid import UUID

import pytest
from asr_lab_routes import FreezeDatasetVersionRequest, evaluation_router
from fastapi.routing import APIRoute
from pydantic import ValidationError
from training_routes import CreateTrainingRunRequest, training_router

_DATASET_ID = UUID("00000000-0000-0000-0000-000000000001")
_VERSION_ID = UUID("00000000-0000-0000-0000-000000000002")
_MODEL_ID = UUID("00000000-0000-0000-0000-000000000003")


def _request_values() -> dict[str, object]:
    return {
        "base_model_version_id": _MODEL_ID,
        "candidate_model_name": "qwen3-asr-domain-v1",
        "idempotency_key": "training-run-1",
    }


def test_training_request_defaults_to_auto_snapshot_without_validation() -> None:
    request = CreateTrainingRunRequest.model_validate({**_request_values(), "dataset_id": _DATASET_ID})

    assert request.dataset_id == _DATASET_ID
    assert request.dataset_version_id is None
    assert request.run_validation is False


@pytest.mark.parametrize(
    "dataset_values",
    [
        {},
        {"dataset_id": _DATASET_ID, "dataset_version_id": _VERSION_ID},
    ],
)
def test_training_request_requires_exactly_one_dataset_reference(dataset_values: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        CreateTrainingRunRequest.model_validate({**_request_values(), **dataset_values})


def test_training_router_exposes_delete_run_endpoint() -> None:
    assert any(
        route.path == "/{run_id}" and route.methods is not None and "DELETE" in route.methods
        for route in training_router.routes
        if isinstance(route, APIRoute)
    )


def test_freeze_request_requires_preview_checksum() -> None:
    with pytest.raises(ValidationError):
        FreezeDatasetVersionRequest.model_validate({})

    request = FreezeDatasetVersionRequest.model_validate({"expected_checksum": "a" * 64})

    assert request.expected_checksum == "a" * 64


def test_evaluation_router_exposes_delete_run_endpoint() -> None:
    assert any(
        route.path == "/runs/{run_id}" and route.methods is not None and "DELETE" in route.methods
        for route in evaluation_router.routes
        if isinstance(route, APIRoute)
    )
