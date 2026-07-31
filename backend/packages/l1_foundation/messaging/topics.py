from __future__ import annotations


class Topics:
    PROCESSING_COMMANDS = "processing.commands"
    PROCESSING_EVENTS = "processing.events"
    PROCESSING_STATE = "processing.state"
    PROCESSING_CANCEL = "processing.cancel"
    PROCESSING_CANCEL_RETRY = "processing.cancel.retry"
    PROCESSING_CANCEL_DLQ = "processing.cancel.dlq"
    PROCESSING_RETRY = "processing.retry"
    PROCESSING_DLQ = "processing.dlq"

    COMPUTE_TASKS_IO = "compute.tasks.io"
    COMPUTE_TASKS_CPU = "compute.tasks.cpu"
    COMPUTE_TASKS_GPU_HIGH = "compute.tasks.gpu-high"
    COMPUTE_TASKS_GPU_NORMAL = "compute.tasks.gpu-normal"
    COMPUTE_RESULTS = "compute.results"
    COMPUTE_STATE = "compute.state"
    COMPUTE_CANCEL = "compute.cancel"
    COMPUTE_CANCEL_RETRY = "compute.cancel.retry"
    COMPUTE_CANCEL_DLQ = "compute.cancel.dlq"
    COMPUTE_RETRY = "compute.retry"
    COMPUTE_DLQ = "compute.dlq"

    GENERATION_COMMANDS = "generation.commands"
    GENERATION_EVENTS = "generation.events"
    GENERATION_STATE = "generation.state"
    GENERATION_CANCEL = "generation.cancel"
    GENERATION_CANCEL_RETRY = "generation.cancel.retry"
    GENERATION_CANCEL_DLQ = "generation.cancel.dlq"
    GENERATION_RETRY = "generation.retry"
    GENERATION_DLQ = "generation.dlq"
    GENERATION_PROJECTION_RETRY = "generation.projection.retry"
    GENERATION_PROJECTION_DLQ = "generation.projection.dlq"

    RAG_EXECUTION_EVENTS = "rag.execution-events"
    MODEL_INVOCATION_EVENTS = "model.invocation-events"
    OBSERVABILITY_DLQ = "observability.dlq"
    OBSERVABILITY_RETRY = "observability.retry"

    COMPACTED = (PROCESSING_STATE, COMPUTE_STATE, GENERATION_STATE)
    ALL = (
        PROCESSING_COMMANDS,
        PROCESSING_EVENTS,
        PROCESSING_STATE,
        PROCESSING_CANCEL,
        PROCESSING_CANCEL_RETRY,
        PROCESSING_CANCEL_DLQ,
        PROCESSING_RETRY,
        PROCESSING_DLQ,
        COMPUTE_TASKS_IO,
        COMPUTE_TASKS_CPU,
        COMPUTE_TASKS_GPU_HIGH,
        COMPUTE_TASKS_GPU_NORMAL,
        COMPUTE_RESULTS,
        COMPUTE_STATE,
        COMPUTE_CANCEL,
        COMPUTE_CANCEL_RETRY,
        COMPUTE_CANCEL_DLQ,
        COMPUTE_RETRY,
        COMPUTE_DLQ,
        GENERATION_COMMANDS,
        GENERATION_EVENTS,
        GENERATION_STATE,
        GENERATION_CANCEL,
        GENERATION_CANCEL_RETRY,
        GENERATION_CANCEL_DLQ,
        GENERATION_RETRY,
        GENERATION_DLQ,
        GENERATION_PROJECTION_RETRY,
        GENERATION_PROJECTION_DLQ,
        RAG_EXECUTION_EVENTS,
        MODEL_INVOCATION_EVENTS,
        OBSERVABILITY_DLQ,
        OBSERVABILITY_RETRY,
    )
