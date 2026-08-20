from asr_lab_routes import evaluation_router as asr_evaluation_router
from asr_lab_routes import model_router
from evaluation_auth_routes import router as auth_router
from evaluation_health_routes import router as health_router
from fastapi import APIRouter
from rag_adjudication_evaluation_routes import router as rag_adjudication_evaluation_router
from rag_evaluation_routes import router as rag_evaluation_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(asr_evaluation_router, prefix="/evaluation", tags=["evaluation"])
router.include_router(rag_evaluation_router, prefix="/evaluation/rag", tags=["rag-evaluation"])
router.include_router(
    rag_adjudication_evaluation_router,
    prefix="/evaluation/rag-adjudication",
    tags=["rag-adjudication-evaluation"],
)
router.include_router(model_router, prefix="/model-versions", tags=["model-versions"])
