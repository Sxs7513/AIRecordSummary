from conversations_routes import router as conversations_router
from fastapi import APIRouter
from generations_routes import router as generations_router
from production_auth_routes import router as auth_router
from production_health_routes import router as health_router
from rag_routes import router as rag_router
from recordings_routes import router as recordings_router
from speaker_profiles_routes import router as speaker_profiles_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
router.include_router(recordings_router, prefix="/recordings", tags=["recordings"])
router.include_router(speaker_profiles_router, prefix="/speaker-profiles", tags=["speaker-profiles"])
router.include_router(rag_router, prefix="/rag", tags=["rag"])
router.include_router(generations_router, prefix="/generations", tags=["generations"])
