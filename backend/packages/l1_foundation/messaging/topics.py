from __future__ import annotations


class Topics:
    PROCESSING_COMMANDS = "processing.commands"
    PROCESSING_EVENTS = "processing.events"
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
    COMPUTE_CANCEL = "compute.cancel"
    COMPUTE_CANCEL_RETRY = "compute.cancel.retry"
    COMPUTE_CANCEL_DLQ = "compute.cancel.dlq"
    COMPUTE_RETRY = "compute.retry"
    COMPUTE_DLQ = "compute.dlq"

    GENERATION_COMMANDS = "generation.commands"
    GENERATION_CANCEL = "generation.cancel"
    GENERATION_CANCEL_RETRY = "generation.cancel.retry"
    GENERATION_CANCEL_DLQ = "generation.cancel.dlq"
    GENERATION_RETRY = "generation.retry"
    GENERATION_DLQ = "generation.dlq"

    RAG_EXECUTION_EVENTS = "rag.execution-events"
    MODEL_INVOCATION_EVENTS = "model.invocation-events"
    OBSERVABILITY_DLQ = "observability.dlq"
    OBSERVABILITY_RETRY = "observability.retry"

    ALL = (
        PROCESSING_COMMANDS,
        PROCESSING_EVENTS,
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
        COMPUTE_CANCEL,
        COMPUTE_CANCEL_RETRY,
        COMPUTE_CANCEL_DLQ,
        COMPUTE_RETRY,
        COMPUTE_DLQ,
        GENERATION_COMMANDS,
        GENERATION_CANCEL,
        GENERATION_CANCEL_RETRY,
        GENERATION_CANCEL_DLQ,
        GENERATION_RETRY,
        GENERATION_DLQ,
        RAG_EXECUTION_EVENTS,
        MODEL_INVOCATION_EVENTS,
        OBSERVABILITY_DLQ,
        OBSERVABILITY_RETRY,
    )
