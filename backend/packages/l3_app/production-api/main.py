from __future__ import annotations

from app_factory import create_app

app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
