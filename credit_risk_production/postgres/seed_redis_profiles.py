import os
import sys
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

# Import path for cache module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache.redis_client import upsert_profile, r

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Config
POSTGRES_URL = os.getenv("POSTGRES_URL", f"postgresql+psycopg2://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}://{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}")
TABLE = "data"
SCHEMA = "credit_risk"
TTL_SECONDS = 3600
BATCH_SIZE = 1000
ID_COLUMN = "prospectID" # Define unique identifier

# Load rows
def load_rows() -> pd.DataFrame:
    eng = create_engine(POSTGRES_URL, pool_pre_ping=True)
    with eng.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {SCHEMA}.{TABLE}"), conn)
    logger.info(f"Loaded {len(df)} rows from {SCHEMA}.{TABLE}")
    return df

# Write rows to Redis
def seed_profiles(df: pd.DataFrame) -> int:
    """For each row, write profile with all columns as hash fields in Redis."""
    if ID_COLUMN not in df.columns:
        logger.error(f"ID column '{ID_COLUMN}' not found in DataFrame columns: {df.columns}")

        try:
            df[ID_COLUMN] = df["prospectid"]
        except KeyError:
            logger.error(f"ID column '{ID_COLUMN}' not found in DataFrame columns: {df.columns}")
            return 0

    written = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        customer_id = str(row[ID_COLUMN])

        # Convert the row to a plain dict
        profile = {col: (None if pd.isna(val) else val) for col, val in row.items() if col != ID_COLUMN}
        profile = {k: v for k, v in profile.items() if v is not None}

        # Upsert profile to Redis to hande JSON encoded
        upsert_profile(customer_id, profile, ttl_seconds=TTL_SECONDS)

        # Velocity counter as companion key for rate limiting
        r.incr(f"velocity:{customer_id}:app_seed_1h")
        r.expire(f"velocity:{customer_id}:app_seed_1h", TTL_SECONDS)

        written += 1

        if i % BATCH_SIZE == 0:
            logger.info(f"Processed {i} rows...")

    logger.info(f"Finished processing {written} rows.")
    return written


# Verify Redis connection
def verify(sample: int = 3) -> None:
    keys = list(r.scan_iter(match="profile:*", count=100))
    logger.info(f"Total profiles in Redis: {len(keys)}")

    for k in keys[:sample]:
        ttl = r.ttl(k)
        nfields = r.hlen(k)
        logger.info(f"Sample key: {k.decode()} | TTL: {ttl}s | Fields: {nfields}")

# Main execution
def main():
    if not r.ping():
        logger.error("Cannot connect to Redis. Please check your Redis server.")
        sys.exit(1)

    df = load_rows()
    seed_profiles(df)
    verify()

if __name__ == "__main__":
    main()