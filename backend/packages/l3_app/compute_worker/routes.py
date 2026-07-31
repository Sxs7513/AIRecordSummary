from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from l3_app.compute_worker.runtime import ComputeWorkerMetrics, ComputeWorkerRuntime

router = APIRouter()
InternalToken = Annotated[str | None, Header(alias="X-Internal-Token")]


class WorkerHealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["ok"]


class WorkerReadinessResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["ready"]
    registered_operations: int


class WorkerMetricsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    ready: bool
    registered_operations: int
    total_tasks: int
    queued_tasks: int
    running_tasks: int
    succeeded_tasks: int
    failed_tasks: int
    cancelled_tasks: int

    @classmethod
    def from_metrics(cls, metrics: ComputeWorkerMetrics) -> WorkerMetricsResponse:
        return cls.model_validate(asdict(metrics))


def _runtime(request: Request, token: str | None) -> ComputeWorkerRuntime:
    runtime = getattr(request.app.state, "compute_worker_runtime", None)
    if not isinstance(runtime, ComputeWorkerRuntime) or not runtime.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Compute worker is not ready")
    if not runtime.authorize(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal compute token")
    return runtime


@router.get("/healthz", response_model=WorkerHealthResponse)
def healthz() -> WorkerHealthResponse:
    return WorkerHealthResponse(status="ok")


@router.get("/readyz", response_model=WorkerReadinessResponse)
def readyz(request: Request, internal_token: InternalToken = None) -> WorkerReadinessResponse:
    metrics = _runtime(request, internal_token).metrics()
    return WorkerReadinessResponse(status="ready", registered_operations=metrics.registered_operations)


@router.get("/metrics", response_model=WorkerMetricsResponse)
def metrics(request: Request, internal_token: InternalToken = None) -> WorkerMetricsResponse:
    return WorkerMetricsResponse.from_metrics(_runtime(request, internal_token).metrics())
