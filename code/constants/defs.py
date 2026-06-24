import os

# ---------------------------------------------------------------------------
# OANDA credentials
#
# Credentials must NEVER be hard-coded in version control. They are loaded from
# environment variables, with an optional git-ignored local override file for
# convenient local development.
#
#   1. Preferred: set environment variables
#        OANDA_API_KEY, OANDA_ACCOUNT_ID  (optionally OANDA_URL)
#   2. Local dev: copy constants/defs.example.py -> constants/defs_local.py
#        and fill in your practice-account values. defs_local.py is git-ignored.
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OANDA_API_KEY", "")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_URL = os.environ.get("OANDA_URL", "https://api-fxpractice.oanda.com/v3")


def _load_local_overrides():
    """Import a git-ignored constants/defs_local.py if present."""
    try:
        from constants import defs_local  # type: ignore
        return defs_local
    except Exception:
        try:
            from . import defs_local  # type: ignore
            return defs_local
        except Exception:
            return None


_local = _load_local_overrides()
if _local is not None:
    API_KEY = API_KEY or getattr(_local, "API_KEY", "")
    ACCOUNT_ID = ACCOUNT_ID or getattr(_local, "ACCOUNT_ID", "")
    OANDA_URL = getattr(_local, "OANDA_URL", OANDA_URL)

if not API_KEY or not ACCOUNT_ID:
    # Don't crash on import (e.g. for offline notebooks); warn instead.
    import warnings

    warnings.warn(
        "OANDA_API_KEY / OANDA_ACCOUNT_ID are not set. Set them as environment "
        "variables or create constants/defs_local.py (see constants/defs.example.py).",
        RuntimeWarning,
    )

SELL = -1
BUY = 1
NONE = 0
