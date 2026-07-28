from __future__ import annotations

import ast
from pathlib import Path

from fastapi import APIRouter

from router import evaluation_api_router, production_api_router, training_api_router
from routes.asr_lab import evaluation_router, training_router
from routes.recordings import router as recordings_router

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _includes(parent: APIRouter, child: APIRouter) -> bool:
    return any(vars(route).get("original_router") is child for route in parent.routes)


def test_l3_api_routers_have_separate_route_surfaces() -> None:
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
        "L2-Core/trainers/qwen-asr-lora",
        "L3-App/shared-api",
        "L3-App/production-api",
        "L3-App/evaluation-api",
        "L3-App/training-api",
        "L3-App/production-worker",
        "L3-App/asr-compute-worker",
    )
    for relative_path in expected_packages:
        assert (BACKEND_ROOT / relative_path).is_dir(), relative_path

    assert (BACKEND_ROOT / "pyproject.toml").is_file()
    trainer_project = BACKEND_ROOT / "L2-Core/trainers/qwen-asr-lora/pyproject.toml"
    assert trainer_project.is_file()
    nested_projects = set(BACKEND_ROOT.glob("packages/*/pyproject.toml"))
    nested_projects.update(BACKEND_ROOT.glob("L3-App/*/pyproject.toml"))
    assert nested_projects == set()


def test_l3_app_sources_are_directly_under_src() -> None:
    l3_root = BACKEND_ROOT / "L3-App"
    for app_dir in l3_root.iterdir():
        if not app_dir.is_dir():
            continue
        source_root = app_dir / "src"
        assert source_root.is_dir(), app_dir
        assert not any(path.name.startswith("airecord_") for path in source_root.iterdir() if path.is_dir())


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
        BACKEND_ROOT / "L3-App",
        BACKEND_ROOT / "scripts",
        BACKEND_ROOT / "tests",
    )
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
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
    assert (BACKEND_ROOT / "L2-Core/trainers/qwen-asr-lora/src/airecord_qwen_asr_trainer/trainer.py").is_file()


def test_legacy_src_contains_no_python_source() -> None:
    legacy_src = BACKEND_ROOT / "src"
    assert not list(legacy_src.rglob("*.py"))
