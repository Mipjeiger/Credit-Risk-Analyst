import sqlite3
import json
from pathlib import Path

# Set path to the SQLite file in local chroma directory
DB_PATH = Path(__file__).resolve().parents[1] / "credit_risk_production" / "database" / "LLM" / "chroma_store" / "chroma.sqlite3"

if not DB_PATH.exists():
    print(f"❌ Error: Database file not found at {DB_PATH.resolve()}")
    exit(1)

print(f"Connecting to {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT id, config_json_str FROM collections;")
rows = cursor.fetchall()

updated_count = 0
for col_id, config_str in rows:
    if config_str:
        config = json.loads(config_str)
        if "_type" not in config:
            config["_type"] = "CollectionConfigurationInternal"
            updated_str = json.dumps(config)
            cursor.execute(
                "UPDATE collections SET config_json_str = ? WHERE id = ?;",
                (updated_str, col_id)
            )
            updated_count += 1

conn.commit()
conn.close()

print(f"Done! Successfully updated {updated_count} collection(s).")