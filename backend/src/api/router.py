from fastapi import APIRouter

from api.routes.asr_lab import evaluation_router, model_router, training_router
from api.routes.auth import router as auth_router
from api.routes.conversations import router as conversations_router
from api.routes.generations import router as generations_router
from api.routes.health import router as health_router
from api.routes.rag import router as rag_router
from api.routes.recordings import router as recordings_router
from api.routes.speaker_profiles import router as speaker_profiles_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
api_router.include_router(recordings_router, prefix="/recordings", tags=["recordings"])
api_router.include_router(speaker_profiles_router, prefix="/speaker-profiles", tags=["speaker-profiles"])
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(generations_router, prefix="/generations", tags=["generations"])
api_router.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
api_router.include_router(training_router, prefix="/training-runs", tags=["training-runs"])
api_router.include_router(model_router, prefix="/model-versions", tags=["model-versions"])
