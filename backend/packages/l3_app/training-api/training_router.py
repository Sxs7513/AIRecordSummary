from fastapi import APIRouter
from training_auth_routes import router as auth_router
from training_health_routes import router as health_router
from training_routes import model_router, training_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(training_router, prefix="/training-runs", tags=["training-runs"])
router.include_router(model_router, prefix="/model-versions", tags=["model-versions"])
