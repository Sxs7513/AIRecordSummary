# Backend package migration

`packages/l1_foundation`, `packages/l2_core`, and `L3-App` are now the
canonical package and entry-point boundaries.

L1/L2/L3 are logical packages managed by the root `backend/pyproject.toml`;
they are not separately published Python projects. The Qwen HF Trainer is the
only nested project with its own `pyproject.toml` and `.venv`, because its
Transformers dependency conflicts with production inference.

All Python implementations have been physically moved out of `backend/src`.
L1 and L2 packages live under the single `backend/packages` import root as
`l1_foundation` and `l2_core`. Each L3 application keeps its own direct `src`
entry point.

The temporary `airecord_*` facade packages have been removed. Runtime code
uses explicit layered namespaces such as `l1_foundation.pipeline`,
`l2_core.audio_processing`, `l2_core.rag` and `l2_core.conversations`.
`backend/src` contains no Python source and is not part of the runtime path.
