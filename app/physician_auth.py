# -*- coding: utf-8 -*-
"""
app/physician_auth.py

Server-side authentication utilities for the physician review portal.

DESIGN:
  - No hard-coded credentials.
  - Credentials come exclusively from environment variables:
      REVIEWER_USERNAME         — login username
      REVIEWER_PASSWORD_HASH    — PBKDF2-SHA256 hash produced by hash_password()
      REVIEW_SESSION_SECRET     — signing key for Starlette SessionMiddleware
  - Password hashing: stdlib hashlib.pbkdf2_hmac (no external bcrypt dep).
  - Hash format: "pbkdf2sha256$<iterations>$<hex-salt>$<hex-digest>"
    (compatible with generate_hash CLI helper below).
  - Sessions: Starlette signed-cookie sessions (itsdangerous, transitive dep).
  - Cookie flags: HttpOnly=True, SameSite=lax, Secure when behind HTTPS proxy.

HELPER (run locally to generate REVIEWER_PASSWORD_HASH):
  python -c "from app.physician_auth import generate_hash; print(generate_hash('your-password'))"
"""

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PBKDF2_ITERATIONS = 260_000   # OWASP 2023 recommendation for PBKDF2-SHA256
_SALT_BYTES = 32
_HASH_PREFIX = "pbkdf2sha256"

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def generate_hash(password: str) -> str:
    """
    Hash a plaintext password.  Call this once to populate REVIEWER_PASSWORD_HASH.

    Example (run from project root):
        python -c "from app.physician_auth import generate_hash; print(generate_hash('s3cr3t!'))"
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{_HASH_PREFIX}${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """
    Constant-time verification of plaintext against a stored hash string.
    Returns False (never raises) on any format error.
    """
    try:
        prefix, iters_str, salt_hex, dk_hex = stored_hash.split("$", 3)
        if prefix != _HASH_PREFIX:
            return False
        iterations = int(iters_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
        computed = hashlib.pbkdf2_hmac(
            "sha256", plaintext.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(computed, expected)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Environment-based credential loading
# ---------------------------------------------------------------------------


def _get_reviewer_username() -> Optional[str]:
    return os.environ.get("REVIEWER_USERNAME", "").strip() or None


def _get_reviewer_password_hash() -> Optional[str]:
    return os.environ.get("REVIEWER_PASSWORD_HASH", "").strip() or None


_MIN_SECRET_BYTES = 32   # 256-bit minimum for HMAC signing key

# Placeholder values that must never pass as a real secret
_PLACEHOLDER_SECRETS: frozenset = frozenset({
    "changeme", "secret", "your-secret", "your_secret",
    "xxx", "placeholder", "replace-me", "replace_me",
    "<random-hex-string>", "yoursecretkey",
})


def _is_strong_secret(raw: str) -> bool:
    """Return True only when raw is a non-empty, non-placeholder secret of adequate length."""
    stripped = raw.strip()
    if not stripped:
        return False
    if len(stripped) < _MIN_SECRET_BYTES:
        return False
    if stripped.lower() in _PLACEHOLDER_SECRETS:
        return False
    return True


def portal_enabled() -> bool:
    """
    True only when all required env vars are present AND the session secret
    passes minimum-strength validation.  A missing, short, or placeholder
    secret disables the portal while leaving the patient chat unaffected.
    """
    enabled_flag = os.environ.get("REVIEW_PORTAL_ENABLED", "false").lower()
    if enabled_flag not in ("1", "true", "yes"):
        return False
    username = _get_reviewer_username()
    pw_hash = _get_reviewer_password_hash()
    raw_secret = os.environ.get("REVIEW_SESSION_SECRET", "")
    if not _is_strong_secret(raw_secret):
        logger.warning(
            "physician_auth: portal disabled — REVIEW_SESSION_SECRET is missing, "
            "too short (< %d chars), or a placeholder value.  "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"",
            _MIN_SECRET_BYTES,
        )
        return False
    return bool(username and pw_hash)


def get_session_secret() -> str:
    """Return REVIEW_SESSION_SECRET or a random fallback (dev only)."""
    secret = os.environ.get("REVIEW_SESSION_SECRET", "").strip()
    if secret:
        if len(secret) < _MIN_SECRET_BYTES:
            logger.warning(
                "physician_auth: REVIEW_SESSION_SECRET is only %d chars — "
                "use at least %d chars (generate with: "
                "python -c \"import secrets; print(secrets.token_hex(32))\")",
                len(secret), _MIN_SECRET_BYTES,
            )
        return secret
    # Development fallback — logs a warning so operators notice
    logger.warning(
        "physician_auth: REVIEW_SESSION_SECRET not set — using ephemeral random key. "
        "All sessions will be invalidated on each restart."
    )
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_SESSION_KEY = "physician_session"
_MAX_SESSION_AGE_SECONDS = 8 * 3600   # 8 hours

# ---------------------------------------------------------------------------
# Brute-force protection
# ---------------------------------------------------------------------------

_MAX_FAILED_ATTEMPTS = 5          # attempts before lockout
_ATTEMPT_WINDOW_SECONDS = 300     # 5-minute rolling window
_LOCKOUT_SECONDS = 600            # 10-minute lockout after threshold

_failed_attempts: dict = {}       # username → [timestamp, ...]
_lockouts: dict = {}              # username → lockout_until_timestamp
_bf_lock = threading.Lock()


def _record_failed_login(username: str) -> None:
    now = time.monotonic()
    with _bf_lock:
        timestamps: List[float] = _failed_attempts.get(username, [])
        timestamps = [t for t in timestamps if now - t < _ATTEMPT_WINDOW_SECONDS]
        timestamps.append(now)
        _failed_attempts[username] = timestamps
        if len(timestamps) >= _MAX_FAILED_ATTEMPTS:
            _lockouts[username] = now + _LOCKOUT_SECONDS
            logger.warning(
                "physician_auth: login locked out for '%s' after %d failed attempts",
                username, len(timestamps),
            )


def _clear_failed_login(username: str) -> None:
    with _bf_lock:
        _failed_attempts.pop(username, None)
        _lockouts.pop(username, None)


def is_locked_out(username: str) -> bool:
    now = time.monotonic()
    with _bf_lock:
        until = _lockouts.get(username, 0)
        if now < until:
            return True
        if until:
            _lockouts.pop(username, None)
        return False


def authenticate(username: str, password: str) -> bool:
    """
    Check username + password against environment credentials.
    Constant-time on both checks to prevent timing side-channels.
    Brute-force protection: locks out a username after 5 failed attempts in 5 minutes.
    """
    if is_locked_out(username):
        logger.warning("physician_auth: login attempt from locked-out username '%s'", username)
        return False

    expected_username = _get_reviewer_username()
    stored_hash = _get_reviewer_password_hash()

    if not expected_username or not stored_hash:
        logger.warning("physician_auth: credentials not configured")
        return False

    # Username comparison (case-sensitive, constant-time)
    username_ok = hmac.compare_digest(username, expected_username)
    password_ok = verify_password(password, stored_hash)

    # Evaluate both before returning to avoid short-circuit timing leak
    success = username_ok and password_ok
    if success:
        _clear_failed_login(username)
    else:
        _record_failed_login(username)
    return success


def create_physician_session(request, identity: str) -> None:
    """Stamp the Starlette session with physician identity + timestamp."""
    import time
    request.session[_SESSION_KEY] = {
        "identity": identity,
        "issued_at": int(time.time()),
    }


def get_physician_identity(request) -> Optional[str]:
    """
    Return the physician username from the session, or None if invalid/expired.
    """
    import time
    data = request.session.get(_SESSION_KEY)
    if not data:
        return None
    issued_at = data.get("issued_at", 0)
    if time.time() - issued_at > _MAX_SESSION_AGE_SECONDS:
        invalidate_physician_session(request)
        return None
    return data.get("identity")


def invalidate_physician_session(request) -> None:
    """Clear the physician session (logout)."""
    request.session.pop(_SESSION_KEY, None)


def require_physician(request):
    """
    Dependency / guard.  Raises HTTP 401 if not authenticated.
    Use in route handlers before doing any physician work.
    """
    from fastapi import HTTPException
    identity = get_physician_identity(request)
    if not identity:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return identity
