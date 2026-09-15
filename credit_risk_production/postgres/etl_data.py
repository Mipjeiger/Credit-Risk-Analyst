import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

"""ETL: Load .parqyet database to PostgreSQL table credit_risk"""

# Configuration
BASE_PATH = Path(__file__).resolve().parents[2]
PROJECT = BASE_PATH / "credit_risk_production"
PARQUET_PATH = PROJECT / "database" / "data" / "merged_credit_risk_data.parquet"
SQL_PATH = Path(__file__).resolve().parent / "credit_risk.sql"

# Connection configuration
ENV_PATH = PROJECT / ".env"
load_dotenv(ENV_PATH)

# PostgreSQL connection parameters from environment variables
PG_HOST = "localhost"  # Default to localhost if not set
PG_PORT = os.getenv("POSTGRES_PORT")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_DB = os.getenv("POSTGRES_DB")

# Table & schema configuration
SCHEMA_NAME = "credit_risk"
TABLE_NAME = f"{SCHEMA_NAME}.data"

# Optional: features configuration
ROW_LIMIT = int(os.getenv("ROW_LIMIT", 12000))  # Default to 12000 if not set
APPLY_SCHEMA = os.getenv("APPLY_SCHEMA", "false").lower() == "true"

# Build PostgreSQL connection string
def get_engine() -> Engine:
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url, pool_pre_ping=True)

def sql_column_name(col: str) -> str:
    return col.strip().lower()

def load_parquet(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the parquet file, optionally limitting to the first nrows."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    if nrows is not None:
        df = df.head(nrows)

    # Normalie column names to match the SQL table schema
    df.columns = [sql_column_name(col) for col in df.columns]
    return df

def apply_schema(engine: Engine) -> None:
    """Create schema and table if they do not exist. (idempotent)"""
    ddl = SQL_PATH.read_text()
    with engine.begin() as conn:
        conn.execute(text(ddl))
        print(f"Schema and table created or already exist: {TABLE_NAME}")

def load_to_postgres(engine: Engine, df: pd.DataFrame) -> int:
    if "prospectid" not in df.columns:
        raise KeyError("Expected 'prospectid' column.")

    staging = "data_staging"

    # Stage the table to schema & table name
    df.to_sql(
        name=staging,
        con=engine,
        schema=SCHEMA_NAME,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000
    )

    # Insert into main table, avoiding duplicates
    cols = ", ".join(df.columns)
    with engine.begin() as conn:
        conn.execute(
            text(f"""
            INSERT INTO {TABLE_NAME} ({cols})
            SELECT {cols} FROM {SCHEMA_NAME}.{staging}
            ON CONFLICT (prospectid) DO NOTHING
        """)
        )
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA_NAME}.{staging}"))

    return len(df)

# Main ETL function
def run() -> None:
    print(f"Reading {PARQUET_PATH} with row limit {ROW_LIMIT}...")
    df = load_parquet(PARQUET_PATH, nrows=ROW_LIMIT)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns.")

    engine = get_engine()

    # Run schema migration if enabled
    if APPLY_SCHEMA:
        print(f"Applying schema from {SQL_PATH}")
        apply_schema(engine)
    else:
        print("→ Skipping apply_schema (AUTO_MIGRATE=false)")

    print(f"Loading data into PostgreSQL table {TABLE_NAME}...")
    n = load_to_postgres(engine, df)
    print(f"✅ Attempted {n:,} rows into {TABLE_NAME}.")

    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        print(f"Total rows in {TABLE_NAME}: {total:,}")

if __name__ == "__main__":
    run()