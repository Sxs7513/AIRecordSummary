from __future__ import annotations

from app_factory import create_app
from router import production_api_router

app = create_app(router=production_api_router)


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
