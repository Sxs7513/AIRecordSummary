#!/usr/bin/env python3
import contextlib
import json
import re
import sys


def protect_terms(text: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    protected = text
    placeholders: dict[str, str] = {}
    for index, term in enumerate(sorted(set(terms), key=len, reverse=True)):
        if not term or term not in protected:
            continue
        placeholder = f"ZXQPROTECTEDTERM{index:04d}QXZ"
        placeholders[placeholder] = term
        protected = protected.replace(term, placeholder)
    return protected, placeholders


def restore_terms(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for placeholder, term in placeholders.items():
        restored = restored.replace(placeholder, term)
    return restored


def resolve_correct_function(pycorrector):
    if hasattr(pycorrector, "correct"):
        return pycorrector.correct

    candidate_modules = [
        "pycorrector.corrector",
        "pycorrector.macbert.macbert_corrector",
    ]
    for module_name in candidate_modules:
        try:
            module = __import__(module_name, fromlist=["Corrector", "MacBertCorrector"])
            corrector_class = getattr(module, "Corrector", None) or getattr(module, "MacBertCorrector", None)
            if corrector_class is None:
                continue
            corrector = corrector_class()
            if hasattr(corrector, "correct"):
                return corrector.correct
        except Exception:
            continue

    return None


def correct_one(correct, text: str, protected_terms: list[str]) -> str:
    if not text.strip():
        return text

    protected, placeholders = protect_terms(text, protected_terms)
    result = correct(protected)
    if isinstance(result, tuple):
        corrected = result[0]
    elif isinstance(result, dict):
        corrected = result.get("target") or result.get("corrected_text") or result.get("text") or protected
    else:
        corrected = str(result)

    corrected = restore_terms(corrected, placeholders)
    corrected = re.sub(r"\s+", " ", corrected).strip() if re.search(r"[A-Za-z]", corrected) else corrected
    return corrected


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    texts = payload.get("texts") or []
    protected_terms = payload.get("protectedTerms") or []
    if not isinstance(texts, list):
        raise RuntimeError("texts must be a list")

    with contextlib.redirect_stdout(sys.stderr):
        try:
            import pycorrector
        except Exception as exc:
            raise RuntimeError("pycorrector is not installed. Run scripts/install_audio_dependencies.sh first.") from exc

    correct = resolve_correct_function(pycorrector)
    if correct is None:
        print("pycorrector does not expose a supported correct API; returning original texts.", file=sys.stderr)
        print(json.dumps({"texts": [str(text) for text in texts]}, ensure_ascii=False))
        return 0

    try:
        corrected = [correct_one(correct, str(text), [str(term) for term in protected_terms]) for text in texts]
    except Exception as exc:
        print(f"pycorrector unavailable in this environment: {exc}; returning original texts.", file=sys.stderr)
        corrected = [str(text) for text in texts]
    print(json.dumps({"texts": corrected}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
