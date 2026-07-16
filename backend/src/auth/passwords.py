from __future__ import annotations

import base64
import hashlib
import hmac
import os

_N = 2**14
_R = 8
_P = 1
_KEY_LENGTH = 32


def hash_password(password: str) -> str:
    """Return a self-describing scrypt password hash."""
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    salt = os.urandom(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_KEY_LENGTH)
    return "$".join(("scrypt", str(_N), str(_R), str(_P), _encode(salt), _encode(derived)))


def verify_password(password: str, encoded_hash: str) -> bool:
    """Compare a password with an application-generated scrypt hash."""
    try:
        algorithm, n, r, p, encoded_salt, expected = encoded_hash.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(password.encode("utf-8"), salt=_decode(encoded_salt), n=int(n), r=int(r), p=int(p), dklen=_KEY_LENGTH)
        return hmac.compare_digest(actual, _decode(expected))
    except ValueError, TypeError:
        return False


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
