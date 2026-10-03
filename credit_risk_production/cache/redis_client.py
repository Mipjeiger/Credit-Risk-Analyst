import os
import json
import numpy as np
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# decode responses from Redis as UTF-8 strings
r = redis.from_url(REDIS_URL, decode_responses=False)

# ------- Identify & Behavioral Profiling -------
def upsert_profile(customer_id: str, profile: dict, ttl_seconds: int = 86400):
    """profile: {age, income, device_id, txn_count_24h, avg_ticket, ...}"""
    key = f"profile:{customer_id}"
    
    # Store the profile as a hash in Redis, with JSON-encoded values for dicts/lists
    r.hset(key, mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in profile.items()})
    r.expire(key, ttl_seconds) 

def get_profile(customer_id: str) -> dict | None:
    data = r.hgetall(f"profile:{customer_id}")
    if not data:
        return None
    return {k.decode(): v.decode() for k, v in data.items()}

# ------- Velocity / Space-efficient Screening -------
def incr_velocity(customer_id: str, bucket: str, ttl: int = 3600) -> float:
    """bucket examples: txn_1h, login_1h, apps_24h. Returns new count."""
    key = f"velocity:{customer_id}:{bucket}"
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, ttl)
    return float(pipe.execute()[0])

# ------- Amount based velocity amounting -------
def add_velocity_amount(customer_id: str, bucket: str, amount: float, ttl: int = 3600) -> float:
    """bucket examples: txn_amt_1h, login_amt_1h, apps_amt_24h. Returns new total amount."""
    key = f"velocity:{customer_id}:{bucket}"
    pipe = r.pipeline()
    pipe.incrbyfloat(key, amount)
    pipe.expire(key, ttl)
    return float(pipe.execute()[0])

def compute_screen_bits(profile: dict) -> int:
    """Pack risk flags into one integer."""
    bits = 0
    try:
        if float(profile.get("NETMONTHLYINCOME", 0) or 0) < 20_000:
            bits |= 1 << 0  # low income
        if int(profile.get("num_deliq_6mts", 0) or 0) > 0:
            bits |= 1 << 1  # high delinquency
        if float(profile.get("Unsecured_TL", 0) or 0) / max(float(profile.get("Total_TL", 1) or 1), 1) > 0.7:
            bits |= 1 << 2  # high unsecured debt ratio
        if str(profile.get("last_prod_enq2", "")).lower() == "al":
            bits |= 1 << 3  # recent AL inquiry

    except (TypeError, ValueError):
        pass

    return bits

def upsert_screen(customer_id: str, profile: dict, ttl: int = 3600) -> int:
    bits = compute_screen_bits(profile)
    r.setex(f"screen:{customer_id}", ttl, bits)
    return bits

def read_screen(customer_id: str) -> int:
    return int(r.get(f"screen:{customer_id}") or 0)

# ------ Device / card graph --------
def register_device(customer_id: str, device_id: str, ttl: int = 7 * 86400) -> int:
    key = f"device:{device_id}:customers"
    pipe = r.pipeline()
    pipe.sadd(key, customer_id)
    pipe.expire(key, ttl)
    return int(pipe.execute()[0])

def check_device_sharing(device_id: str) -> int:
    return int(r.scard(f"device:{device_id}:customers"))

# ------- Vector Search similarity for Fraud -------
VECTOR_DIM = 47 # Match it with features
INDEX_NAME = "idx:fraud_vector"

def ensure_vector_index():
    try:
        r.ft(INDEX_NAME).info()
        return
    except redis.ResponseError:
        pass

    # Schema for vector search
    schema = (
        TextField("customer_id"),
        TextField("label"),
        VectorField("vec", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE",
        }),
    )
    r.ft(INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(prefix=["fraudvec:"], index_type=IndexType.HASH),
    )

def upsert_fraud_vector(customer_id: str, vec: np.ndarray, label: str = "unknown"):
    """Store the fraud vector for a customer in Redis."""
    key = f"fraudvec:{customer_id}"
    r.hset(key, mapping={
        "customer_id": customer_id,
        "label": label,
        "vec": vec.astype(np.float32).tobytes(),
    })

def knn_fraud(vec: np.ndarray, k: int = 5) -> list[dict]:
    """Return the k nearest neighbors to the given fraud vector."""
    q = (f"*=>[KNN {k} @vec $q AS dist]")
    res = r.ft(INDEX_NAME).search(q, query_params={"q": vec.astype(np.float32).tobytes()})
    return [
        {
            "customer_id": doc.customer_id.decode(),
            "label": doc.label.decode(),
            "distance": float(doc.dist),
        }
        for doc in res.docs
    ]