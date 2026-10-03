import json
import time
import logging
from pathlib import Path

from cache.redis_client import r

logger = logging.getLogger(__name__)

# Load the Lua script for exposure once at import time
_LUA_PATH = Path(__file__).resolve().parents[1] / "postgres" / "check_and_draw.lua"
_check_and_draw = None
if _LUA_PATH.exists():
    _check_and_draw = r.register_script(_LUA_PATH.read_text())
else:
    logger.warning("check_and_draw.lua not found at %s", _LUA_PATH)

# ------------------------------------------------------------
# Limits & drawing
# ------------------------------------------------------------
def approve_credit(customer_id: str, new_limit: float) -> float:
    """Automatically increase the approved limit"""
    return float(r.incrbyfloat(f"limit:{customer_id}", new_limit))

def draw_credit(customer_id: str, amount: float) -> float:
    """Atomatically draw from the credit line"""
    if _check_and_draw is None:
        raise RuntimeError("Lua script not loaded, cannot draw credit")

   # Call the Lua script to check and draw atomically
    res = _check_and_draw(
        keys=[f"limit:{customer_id}", f"drawn:{customer_id}"],
        args=[amount],
    )

    if float(res) < 0:
        raise ValueError(f"Insufficient credit for customer {customer_id}")

    return float(res)

def get_exposure(customer_id: str) -> dict:
    pipe = r.pipeline()
    pipe.get(f"limit:{customer_id}")
    pipe.get(f"drawn:{customer_id}")
    limit, drawn = pipe.execute()
    limit = float(limit or 0.0)
    drawn = float(drawn or 0.0)
    return {
        "limit": limit,
        "drawn": drawn,
        "available": limit - drawn,
    }

# ---------------------------------------------------------------------------
# Write-ahead log for exposure changes (survives a Redis restart)
# ---------------------------------------------------------------------------
def append_exposure_wal(customer_id: str, amount: float, new_drawn: float) -> None:
    r.lpush("exposure:wal", json.dumps({
        "cid": customer_id,
        "amount": amount,
        "new_drawn": new_drawn,
        "at": time.time(),
    }))

# ---------------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------------
def record_decision(customer_id: str, limit: float, hour_key: str) -> None:
    pipe = r.pipeline()
    pipe.hincrbyfloat(f"metrics:hourly:{hour_key}", "total_limit", limit)
    pipe.hincrby(f"metrics:hourly:{hour_key}", "n_decisions", 1)
    pipe.expire(f"metrics:hourly:{hour_key}", 48 * 3600)  # Keep metrics for 48 hours
    pipe.execute()

# ----------------------------------------------------------------------------
# Async job queue
# ----------------------------------------------------------------------------
def enqueue_score(customer_id: str, model_name: str, features: dict) -> None:
    """Enqueue a scoring job for async processing."""
    r.lpush("portofolio:queue", json.dumps({
        "cid": customer_id,
        "model": model_name,
        "features": features,
        "enqueued_at": time.time(),
    }))

def drain_queue(timeout: int = 5) -> dict | None:
    """Drain a scoring job from the queue, waiting up to `timeout` seconds."""
    item = r.brpop("portofolio:queue", timeout=timeout)
    if not item:
        return None
    return json.loads(item[1])