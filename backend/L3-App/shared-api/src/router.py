from fastapi import APIRouter

from routes.asr_lab import evaluation_router, model_router, training_router
from routes.auth import router as auth_router
from routes.conversations import router as conversations_router
from routes.generations import router as generations_router
from routes.health import router as health_router
from routes.rag import router as rag_router
from routes.recordings import router as recordings_router
from routes.speaker_profiles import router as speaker_profiles_router

production_api_router = APIRouter()
production_api_router.include_router(health_router, tags=["health"])
production_api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
production_api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
production_api_router.include_router(recordings_router, prefix="/recordings", tags=["recordings"])
production_api_router.include_router(speaker_profiles_router, prefix="/speaker-profiles", tags=["speaker-profiles"])
production_api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
production_api_router.include_router(generations_router, prefix="/generations", tags=["generations"])

evaluation_api_router = APIRouter()
evaluation_api_router.include_router(health_router, tags=["health"])
evaluation_api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
evaluation_api_router.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
evaluation_api_router.include_router(model_router, prefix="/model-versions", tags=["model-versions"])

training_api_router = APIRouter()
training_api_router.include_router(health_router, tags=["health"])
training_api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
training_api_router.include_router(training_router, prefix="/training-runs", tags=["training-runs"])
training_api_router.include_router(model_router, prefix="/model-versions", tags=["model-versions"])

# Backward-compatible aggregate used by the current single-port development command.
api_router = APIRouter()
api_router.include_router(production_api_router)
api_router.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
api_router.include_router(training_router, prefix="/training-runs", tags=["training-runs"])
api_router.include_router(model_router, prefix="/model-versions", tags=["model-versions"])
