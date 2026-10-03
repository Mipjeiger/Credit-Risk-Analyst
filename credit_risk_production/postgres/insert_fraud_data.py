import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT = Path(__file__).resolve().parents[1]
PARQUET_PATH = (
    PROJECT / "database" / "data" / "merged_credit_risk_data_with_fraud.parquet"
)

load_dotenv(PROJECT / ".env")

SCHEMA = "credit_risk"
TABLE = "data"

PARQUET_ID_COLUMN = "PROSPECTID"
POSTGRES_ID_COLUMN = "prospectid"

SOURCE_COLUMN = "Fraud"
TARGET_COLUMN = "fraud"
STAGING_TABLE = "_fraud_staging"

POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = "localhost"
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
)

engine = create_engine(DATABASE_URL)

# 1. Read Parquet and cast Fraud column explicitly to integer (0 or 1)
fraud_data = pd.read_parquet(
    PARQUET_PATH,
    columns=[PARQUET_ID_COLUMN, SOURCE_COLUMN],
).rename(columns={
    PARQUET_ID_COLUMN: POSTGRES_ID_COLUMN,
    SOURCE_COLUMN: TARGET_COLUMN
})

# Convert true/false or boolean to integers (0 and 1)
fraud_data[TARGET_COLUMN] = fraud_data[TARGET_COLUMN].astype(int)

# 2. Reset target column in PostgreSQL to INTEGER
with engine.begin() as conn:
    conn.execute(
        text(
            f'ALTER TABLE "{SCHEMA}"."{TABLE}" '
            f'DROP COLUMN IF EXISTS "{TARGET_COLUMN}"'
        )
    )
    conn.execute(
        text(
            f'ALTER TABLE "{SCHEMA}"."{TABLE}" '
            f'ADD COLUMN "{TARGET_COLUMN}" INTEGER'
        )
    )

# 3. Write data to staging table
fraud_data.to_sql(
    name=STAGING_TABLE,
    con=engine,
    schema=SCHEMA,
    if_exists="replace",
    index=False,
    chunksize=1000,
    method="multi",
)

# 4. Update target table using explicit integer casting (::INTEGER)
with engine.begin() as conn:
    conn.execute(
        text(
            f"""
            UPDATE "{SCHEMA}"."{TABLE}" AS target
            SET "{TARGET_COLUMN}" = staging."{TARGET_COLUMN}"::INTEGER
            FROM "{SCHEMA}"."{STAGING_TABLE}" AS staging
            WHERE target."{POSTGRES_ID_COLUMN}" = staging."{POSTGRES_ID_COLUMN}"
            """
        )
    )

    # Clean up staging table
    conn.execute(
        text(f'DROP TABLE IF EXISTS "{SCHEMA}"."{STAGING_TABLE}"')
    )

print(f'Column "{TARGET_COLUMN}" successfully updated with 0 and 1 values! 🚀')