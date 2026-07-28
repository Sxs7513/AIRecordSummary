from __future__ import annotations

from threading import Lock

# The embedded pipeline workers and the RAG executor share one Python process.
# llama.cpp/Metal inference is serialized here to avoid competing model contexts.
local_llm_inference_lock = Lock()
