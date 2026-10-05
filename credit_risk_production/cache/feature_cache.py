import json
import hashlib
import math
from cache.redis_client import r

def _sanitize(obj):
    """Recursively replace NaN / Inf with None so can produces valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    return obj

def _json_default(o):
    """Handle numpy scalars and arrays."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            f = float(o)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"Not JSON serializable: {type(o)}")

def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

def get_or_score(prefix: str, payload: dict, scorer, ttl: int = 300) -> dict:
    """Get the cached score for payload, or compute it using scorer and cache it."""
    key = f"{prefix}:{_hash(payload)}"
    cached = r.get(key)

    if cached:
        try:
            # Decode bytes to string before parsing JSON
            cached_str = cached.decode("utf-8") if isinstance(cached, bytes) else cached
            return json.loads(cached_str)
        except Exception:
            pass

    # Compute the score using the provided scorer function
    result = scorer(payload)
    r.setex(key, ttl, json.dumps(_sanitize(result), default=_json_default))
    return result