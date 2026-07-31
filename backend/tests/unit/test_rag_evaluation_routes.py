from fastapi.routing import APIRoute

from rag_evaluation_routes import router


def test_rag_evaluation_router_exposes_delete_run_endpoint() -> None:
    assert any(
        route.path == "/runs/{run_id}" and route.methods is not None and "DELETE" in route.methods
        for route in router.routes
        if isinstance(route, APIRoute)
    )


def test_rag_evaluation_router_exposes_archive_case_endpoint() -> None:
    assert any(
        route.path == "/cases/{case_id}:archive" and route.methods is not None and "POST" in route.methods
        for route in router.routes
        if isinstance(route, APIRoute)
    )
