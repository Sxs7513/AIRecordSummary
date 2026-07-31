from __future__ import annotations

from l1_foundation.settings import get_settings
from l3_app.compute_worker.app_factory import create_worker_app

app = create_worker_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.compute_worker_host, port=settings.compute_worker_port, access_log=False, workers=1)
