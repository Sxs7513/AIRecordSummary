from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast


class PycorrectorCallable(Protocol):
    def __call__(self, text: str) -> object: ...


def correct_texts_with_pycorrector(texts: Sequence[str], protected_terms: Sequence[str]) -> list[str]:
    """Apply the project's pycorrector pass while preserving configured terms verbatim."""
    correct = _load_corrector()
    return [_correct_one(correct, text, protected_terms) for text in texts]


def _load_corrector() -> PycorrectorCallable:
    try:
        module = import_module("pycorrector")
    except ImportError as error:
        raise RuntimeError("pycorrector is not installed; start the GPU worker with backend/.venv") from error
    direct = getattr(module, "correct", None)
    if callable(direct):
        return cast(PycorrectorCallable, direct)
    for module_name, class_names in (
        ("pycorrector.corrector", ("Corrector",)),
        ("pycorrector.macbert.macbert_corrector", ("MacBertCorrector",)),
    ):
        try:
            candidate_module = import_module(module_name)
        except ImportError:
            continue
        for class_name in class_names:
            corrector_class = getattr(candidate_module, class_name, None)
            if not callable(corrector_class):
                continue
            instance = cast(Callable[[], object], corrector_class)()
            method = getattr(instance, "correct", None)
            if callable(method):
                return cast(PycorrectorCallable, method)
    raise RuntimeError("pycorrector does not expose a supported correct API")


def _correct_one(correct: PycorrectorCallable, text: str, protected_terms: Sequence[str]) -> str:
    if not text.strip():
        return text
    protected, placeholders = _protect_terms(text, protected_terms)
    result = correct(protected)
    if isinstance(result, tuple):
        typed_result = cast(tuple[object, ...], result)
        corrected = str(typed_result[0]) if typed_result else protected
    elif isinstance(result, dict):
        payload = cast(dict[str, object], result)
        corrected = str(payload.get("target") or payload.get("corrected_text") or payload.get("text") or protected)
    else:
        corrected = str(result)
    restored = _restore_terms(corrected, placeholders)
    return re.sub(r"\s+", " ", restored).strip() if re.search(r"[A-Za-z]", restored) else restored


def _protect_terms(text: str, terms: Sequence[str]) -> tuple[str, dict[str, str]]:
    protected = text
    placeholders: dict[str, str] = {}
    for index, term in enumerate(sorted(set(terms), key=len, reverse=True)):
        if term and term in protected:
            placeholder = f"ZXQPROTECTEDTERM{index:04d}QXZ"
            placeholders[placeholder] = term
            protected = protected.replace(term, placeholder)
    return protected, placeholders


def _restore_terms(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for placeholder, term in placeholders.items():
        restored = restored.replace(placeholder, term)
    return restored
