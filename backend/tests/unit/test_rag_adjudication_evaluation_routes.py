from fastapi.routing import APIRoute
from rag_adjudication_evaluation_routes import router


def test_adjudication_router_exposes_phase_one_endpoints() -> None:
    routes = {(route.path, method) for route in router.routes if isinstance(route, APIRoute) and route.methods is not None for method in route.methods}
    assert ("/datasets", "POST") in routes
    assert ("/chunks", "GET") in routes
    assert ("/cases/{case_id}", "PATCH") in routes
    assert ("/cases/{case_id}/evidence", "POST") in routes
    assert ("/evidence/{evidence_id}/corrections", "POST") in routes
    assert ("/corrections/{correction_id}", "PATCH") in routes
    assert ("/datasets/{dataset_id}/versions:freeze", "POST") in routes
    assert ("/runs", "POST") in routes
