from __future__ import annotations

from app_factory import create_app

app = create_app()


def run() -> None:
    import uvicorn

    settings = app.state.settings
    uvicorn.run(app, host=settings.observability_api_host, port=settings.observability_api_port, access_log=False)
