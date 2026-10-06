from __future__ import annotations

import os
import json
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

"""
Bootstrap Superset for credit-risk monitoring.

- Creates/updates the SQL views in Postgres.
- Registers a database, datasets, charts and a dashboard via the Superset API.
- Idempotent: rerun to update chart/dashboard definitions.

Env (from data_science/superset/.env):
    SUPERSET_URL          e.g. http://superset:8088
    SUPERSET_USER         ${SUPERSET_USER}
    SUPERSET_PASSWORD     ${SUPERSET_PASSWORD}
    SUPERSET_DB_NAME      ${SUPERSET_DB_NAME}
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
"""

# ---------------------------------------
# Config
# ---------------------------------------
BASE_PATH = Path(__file__).resolve().parent
ENV_PATH = BASE_PATH / ".env"
load_dotenv(ENV_PATH)

SUPERSET_URL = os.getenv("SUPERSET_URL")
SUPERSET_USER = os.getenv("SUPERSET_USER")
SUPERSET_PASSWORD = os.getenv("SUPERSET_PASSWORD")
SUPERSET_DB_NAME = os.getenv("SUPERSET_DB_NAME")

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST_LOCAL = os.getenv("POSTGRES_HOST_LOCAL", "localhost")
POSTGRES_HOST_SUPERSET = os.getenv("POSTGRES_HOST_SUPERSET", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

SQL_VIEWS_PATH = BASE_PATH / "monitoring_views.sql"
CHARTS_YAML    = BASE_PATH / "charts.yaml"

# Url divided by 2 in the local machine & in the superset container docker service
POSTGRES_URL_LOCAL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST_LOCAL}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
POSTGRES_URL_SUPERSET = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST_SUPERSET}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ---------------------------------------
# 1. Apply SQL views to Postgres
# ---------------------------------------
def apply_views() -> None:
    ddl = SQL_VIEWS_PATH.read_text()
    engine = create_engine(POSTGRES_URL_LOCAL, pool_pre_ping=True)

    # Split queries by semicolon to execute each CREATE VIEW independently
    statements = [stmt.strip() for stmt in ddl.split(";") if stmt.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"✅ applied views from {SQL_VIEWS_PATH.name}")

# ---------------------------------------
# 2. Superset API clients
# ---------------------------------------
class SupersetClient:
    def __init__(self, base_url: str, user: str, password: str):
        self.base = base_url
        self.sess = requests.Session()
        self._login(user, password)

    def _login(self, user: str, password: str) -> None:
        r = self.sess.post(
            f"{self.base}/api/v1/security/login",
            json={"username": user, "password": password,
                  "provider": "db", "refresh": True},
        )
        r.raise_for_status()
        token = r.json()["access_token"]
        self.sess.headers.update({"Authorization": f"Bearer {token}"})

        # CSRF token needed for POST/PUT
        csrf = self.sess.get(f"{self.base}/api/v1/security/csrf_token/").json()
        self.sess.headers.update({"X-CSRFToken": csrf["result"]})

    # -- generic helpers --
    def get(self, path: str, **kw):
        r = self.sess.get(f"{self.base}{path}", **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: dict):
        r = self.sess.post(f"{self.base}{path}", json=payload)
        r.raise_for_status()
        return r.json()

    def put(self, path: str, payload: dict):
        r = self.sess.put(f"{self.base}{path}", json=payload)
        r.raise_for_status()
        return r.json()

    # -- databases --
    def find_database(self, name: str):
        data = self.get("/api/v1/database/", params={"q": json.dumps({"filters": [{"col": "database_name", "opr": "eq", "value": name}]})})
        return (data.get("result") or [None])[0]

    def ensure_database(self, name: str, sqlalchemy_uri: str) -> int:
        existing = self.find_database(name)

        if existing:
            print(f"✅ database '{name}' already exists (id={existing['id']})")
            return existing["id"]

        # Create new database
        payload = {
            "database_name": name,
            "sqlalchemy_uri": sqlalchemy_uri,
            "expose_in_sqllab": True,
            "allow_run_async": False,
            "allow_ctas": False,
            "allow_cvas": False,
            "allow_dml": False,
            "extra": json.dumps({
                "metadata_params": {},
            "engine_params": {},
            "metadata_cache_timeout": {},
            "schemas_allowed_for_csv_upload": ["credit_risk"]
            })
        }

        try:
            created = self.post("/api/v1/database/", payload)
            print(f"✅ created database '{name}' (id={created['id']})")
            return created["id"]
        except requests.exceptions.HTTPError as e:
            print(f"❌ Superset 422 Response Error details: {e.response.text}")
            raise e

    # -- datasets --
    def find_dataset(self, db_id: int, schema: str, table: str) -> int:
        data = self.get("/api/v1/dataset/", params={"q": json.dumps({
            "filters": [
                {"col": "schema", "opr": "eq", "value": schema},
                {"col": "table_name", "opr": "eq", "value": table},
            ]
        })})
        return (data.get("result") or [None])[0]

    def ensure_dataset(self, db_id: int, schema: str, table: str) -> int:
        existing = self.find_dataset(db_id, schema, table)

        if existing:
            print(f"✅ dataset '{schema}.{table}' already exists (id={existing['id']})")
            return existing["id"]

        # Create new dataset
        created = self.post("/api/v1/dataset/", {
            "database": db_id,
            "schema": schema,
            "table_name": table
        })
        print(f"✅ created dataset '{schema}.{table}' (id={created['id']})")
        return created["id"]

    # -- charts --
    def find_chart(self, name: str):
        data = self.get("/api/v1/chart/", params={"q": json.dumps({
            "filters": [{"col": "slice_name", "opr": "eq", "value": name}]})})
        return (data.get("result") or [None])[0]

    def upsert_chart(self, name: str, payload: dict) -> int:
        existing = self.find_chart(name)
        if existing:
            self.put(f"/api/v1/chart/{existing['id']}", payload)
            print(f"↺ updated chart: {name} (id={existing['id']})")
            return existing["id"]
        created = self.post("/api/v1/chart/", payload)
        print(f"✅ created chart: {name} (id={created['id']})")
        return created["id"]

    # -- dashboard --
    def find_dashboard(self, title: str):
        data = self.get("/api/v1/dashboard/", params={"q": json.dumps({
            "filters": [{"col": "dashboard_title", "opr": "eq", "value": title}]})})
        return (data.get("result") or [None])[0]

    def upsert_dashboard(self, title: str, payload: dict) -> int:
        existing = self.find_dashboard(title)
        if existing:
            self.put(f"/api/v1/dashboard/{existing['id']}", payload)
            print(f"↺ updated dashboard: {title} (id={existing['id']})")
            return existing["id"]
        created = self.post("/api/v1/dashboard/", payload)
        print(f"✅ created dashboard: {title} (id={created['id']})")
        return created["id"]

# ------------------------------------------------------------------
# 3. build charts from YAML
# ------------------------------------------------------------------
def build_chart_payload(spec: dict, dataset_id: int) -> dict:
    return {
        "slice_name": spec["name"],
        "viz_type":   spec["viz_type"],
        "datasource_id":   dataset_id,
        "datasource_type": "table",
        "params": json.dumps(spec["params"]),
        "description": spec.get("description", ""),
    }

def build_dashboard_layout(chart_ids: list[int], title: str) -> dict:
    """Standard 2-column layout for Apache Superset v2 dashboard position_json."""
    position_data = {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {
            "type": "ROOT",
            "id": "ROOT_ID",
            "children": ["GRID_ID"]
        },
        "GRID_ID": {
            "type": "GRID",
            "id": "GRID_ID",
            "parents": ["ROOT_ID"],
            "children": []
        }
    }

    # Group charts into 2 column rows
    rows = []
    for i in range(0, len(chart_ids), 2):
        row_id = f"ROW-{i//2}"
        row_charts = chart_ids[i:i+2]

        row_children = []
        for cid in row_charts:
            chart_key = f"CHART-{cid}"
            row_children.append(chart_key)

            # Add CHART node with complete meta payload
            position_data[chart_key] = {
                "type": "CHART",
                "id": chart_key,
                "parents": ["ROOT_ID", "GRID_ID", row_id],
                "children": [],
                "meta": {
                    "chartId": cid,
                    "width": 6,
                    "height": 50,
                    "sliceName": f"Chart {cid}"
                }
            }
            
        # Add ROW node
        position_data[row_id] = {
            "type": "ROW",
            "id": row_id,
            "parents": ["ROOT_ID", "GRID_ID"],
            "children": row_children,
            "meta": {
                "background": "BACKGROUND_TRANSPARENT"
            }
        }
        rows.append(row_id)

    # Attach rows to GRID_ID
    position_data["GRID_ID"]["children"] = rows

    return {
        "dashboard_title": title,
        "published": True,
        "position_json": json.dumps(position_data)
    }

# ------------------------------------------------------------------
# 4. — orchestration
# ------------------------------------------------------------------
def run():
    # 4a. DB views
    apply_views()

    # 4b. Superset
    client = SupersetClient(SUPERSET_URL, SUPERSET_USER, SUPERSET_PASSWORD)
    db_id = client.ensure_database(SUPERSET_DB_NAME, POSTGRES_URL_SUPERSET)

    # 4c. Datasets
    spec = yaml.safe_load(CHARTS_YAML.read_text())
    datasets = {}
    for ds in spec["datasets"]:
        datasets[ds["table"]] = client.ensure_dataset(
            db_id, schema=ds["schema"], table=ds["table"]
        )

    # 4d. Charts
    chart_ids = []
    for chart in spec["charts"]:
        ds_id = datasets[chart["dataset"]]
        cid = client.upsert_chart(chart["name"], build_chart_payload(chart, ds_id))
        chart_ids.append(cid)

    # 4e. Dashboard
    dash_payload = build_dashboard_layout(chart_ids, title=spec["dashboard"]["title"])
    dash_id = client.upsert_dashboard(spec["dashboard"]["title"], dash_payload)
    print(f"\n🎉 dashboard ready: {SUPERSET_URL}/superset/dashboard/{dash_id}/")


if __name__ == "__main__":
    run()