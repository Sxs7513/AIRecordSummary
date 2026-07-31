from __future__ import annotations

import ast
from pathlib import Path

from asr_lab_routes import evaluation_router
from evaluation_auth_routes import router as evaluation_auth_router
from evaluation_router import router as evaluation_api_router
from fastapi import APIRouter
from production_auth_routes import router as production_auth_router
from production_router import router as production_api_router
from recordings_routes import router as recordings_router
from training_auth_routes import router as training_auth_router
from training_router import router as training_api_router
from training_routes import training_router

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _is_repository_source(path: Path) -> bool:
    ignored_parts = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return not any(part in ignored_parts for part in path.parts)


def _includes(parent: APIRouter, child: APIRouter) -> bool:
    return any(vars(route).get("original_router") is child for route in parent.routes)


def test_l3_api_routers_have_separate_route_surfaces() -> None:
    assert _includes(production_api_router, production_auth_router)
    assert _includes(evaluation_api_router, evaluation_auth_router)
    assert _includes(training_api_router, training_auth_router)
    assert _includes(production_api_router, recordings_router)
    assert not _includes(production_api_router, evaluation_router)
    assert not _includes(production_api_router, training_router)
    assert _includes(evaluation_api_router, evaluation_router)
    assert not _includes(evaluation_api_router, training_router)
    assert _includes(training_api_router, training_router)
    assert not _includes(training_api_router, evaluation_router)


def test_expected_layer_projects_exist() -> None:
    expected_packages = (
        "packages/l1_foundation/settings",
        "packages/l1_foundation/llm",
        "packages/l1_foundation/worker",
        "packages/l1_foundation/task_runtime",
        "packages/l1_foundation/pipeline",
        "packages/l1_foundation/infrastructure",
        "packages/l2_core/access",
        "packages/l2_core/application",
        "packages/l2_core/audio_processing",
        "packages/l2_core/rag",
        "packages/l2_core/evaluation",
        "packages/l2_core/asr_lab",
        "packages/l2_core/auth",
        "packages/l2_core/conversations",
        "packages/l2_core/generation",
        "packages/l2_core/trainers/qwen-asr-lora",
        "packages/l3_app/production-api",
        "packages/l3_app/evaluation-api",
        "packages/l3_app/rag_evaluation_worker",
        "packages/l3_app/training-api",
        "packages/l3_app/compute_worker",
    )
    for relative_path in expected_packages:
        assert (BACKEND_ROOT / relative_path).is_dir(), relative_path

    assert (BACKEND_ROOT / "pyproject.toml").is_file()
    trainer_project = BACKEND_ROOT / "packages/l2_core/trainers/qwen-asr-lora/pyproject.toml"
    assert trainer_project.is_file()
    nested_projects = {
        path
        for path in BACKEND_ROOT.glob("packages/**/pyproject.toml")
        if _is_repository_source(path)
    }
    assert nested_projects == {trainer_project}


def test_l3_app_sources_are_directly_under_app_root() -> None:
    l3_root = BACKEND_ROOT / "packages/l3_app"
    for app_dir in l3_root.iterdir():
        if not app_dir.is_dir():
            continue
        assert not (app_dir / "src").exists(), app_dir
        assert (app_dir / "main.py").is_file(), app_dir

    assert not (l3_root / "production-worker").exists()
    assert not (l3_root / "asr-compute-worker").exists()


def test_l1_source_does_not_import_l2_or_l3_packages() -> None:
    for path in (BACKEND_ROOT / "packages/l1_foundation").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imports.extend(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
        assert not any(value.startswith("l2_core") for value in imports), path


def test_l2_llm_callers_use_worker_tasks_instead_of_model_instances() -> None:
    roots = (
        BACKEND_ROOT / "packages/l2_core/audio_processing",
        BACKEND_ROOT / "packages/l2_core/rag",
    )
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "LanguageModel" not in source, path
            assert "create_language_model" not in source, path
            assert ".complete(" not in source or "_worker_client.execute(" in source, path


def test_production_api_does_not_embed_compute_worker_runtime() -> None:
    source = (BACKEND_ROOT / "packages/l3_app/production-api/app_factory.py").read_text(encoding="utf-8")
    assert "l3_app.compute_worker" not in source
    assert "build_compute_worker_runtime" not in source


def test_l3_apps_do_not_import_other_l3_apps() -> None:
    l3_root = BACKEND_ROOT / "packages/l3_app"
    app_names = {path.name for path in l3_root.iterdir() if path.is_dir()}
    for app_dir in (path for path in l3_root.iterdir() if path.is_dir()):
        other_apps = app_names - {app_dir.name}
        for path in app_dir.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert not any(f"l3_app.{name.replace('-', '_')}" in source for name in other_apps), path


def test_pipeline_stage_contract_has_no_resource_queue() -> None:
    pipeline_root = BACKEND_ROOT / "packages/l1_foundation/pipeline"
    for path in pipeline_root.rglob("*.py"):
        assert "resource_queue" not in path.read_text(encoding="utf-8"), path


def test_backend_uses_explicit_layered_import_namespaces() -> None:
    legacy_roots = {
        "access",
        "application",
        "asr_lab",
        "audio_processing",
        "auth",
        "conversations",
        "evaluation",
        "generation",
        "infrastructure",
        "pipeline",
        "rag",
        "settings",
        "task_runtime",
    }
    source_roots = (
        BACKEND_ROOT / "packages",
        BACKEND_ROOT / "scripts",
        BACKEND_ROOT / "tests",
    )
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            if not _is_repository_source(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            ]
            imports.extend(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
            assert not any(value.split(".", maxsplit=1)[0] in legacy_roots for value in imports), path


def test_training_implementation_is_not_a_repository_script() -> None:
    assert not (BACKEND_ROOT / "scripts/train_qwen_asr_lora.py").exists()
    assert (BACKEND_ROOT / "packages/l2_core/trainers/qwen-asr-lora/trainer.py").is_file()
    assert not (BACKEND_ROOT / "packages/l2_core/trainers/qwen-asr-lora/src").exists()


def test_legacy_src_contains_no_python_source() -> None:
    legacy_src = BACKEND_ROOT / "src"
    assert not list(legacy_src.rglob("*.py"))
