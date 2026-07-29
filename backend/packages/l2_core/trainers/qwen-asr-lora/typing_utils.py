from typing import Any


def dynamic_attribute(namespace: object, name: str) -> Any:
    """Return an attribute exposed by a third-party lazy module."""

    return getattr(namespace, name)
