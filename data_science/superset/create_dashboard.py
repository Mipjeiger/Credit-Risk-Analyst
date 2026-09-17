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
POSTGRES_HOST = "localhost"
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

POSTGRES_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
SQL_VIEWS_PATH = BASE_PATH / "monitoring_views.sql"
CHARTS_YAML    = BASE_PATH / "charts.yaml"

# ---------------------------------------
# 1. Apply SQL views to Postgres
# ---------------------------------------
def apply_views() -> None:
    ddl = SQL_VIEWS_PATH.read_text()
    engine = create_engine(POSTGRES_URL, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text(ddl))
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
        created = self.post("/api/v1/database/", {
            "database_name": name,
            "sqlalchemy_uri": sqlalchemy_uri,
            "expose_in_sqllab": True,
            "allow_run_async": False,
            "allow_ctas": False,
            "allow_cvas": False,
            "allow_dml": False,
        })
        print(f"✅ created database '{name}' (id={created['id']})")
        return created["id"]

    # -- datasets --
    def find_dataset(self, db_id: int, schema: str, table: str) -> int:
        data = self.get("/api/v1/dataset", params={"q": json.dumps({
            "filters": [
                {"col": "schema", "opr": "eq", "val": schema},
                {"col": "table_name", "opr": "eq", "val": table},
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
            "filters": [{"col": "slice_name", "opr": "eq", "val": name}]})})
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
            "filters": [{"col": "dashboard_title", "opr": "eq", "val": title}]})})
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
    """Simple 2-column grid; every chart gets a slot."""
    grid = []
    for i, cid in enumerate(chart_ids):
        row, col = divmod(i, 2)
        grid.append({
            "type": "CHART",
            "id": f"CHART-{cid}",
            "children": [],
            "meta": {
                "chartId": cid,
                "width": 6, "height": 50,
                "sliceName": f"chart-{cid}",
            },
        })
    return {
        "dashboard_title": title,
        "published": True,
        "position_json": json.dumps({
            "DASHBOARD_VERSION_KEY": "v2",
            "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
            "GRID_ID": {"type": "GRID", "id": "GRID_ID",
                        "children": [f"ROW-{i}" for i in range((len(grid)+1)//2)],
                        "parents": ["ROOT_ID"]},
            **{f"ROW-{i}": {"type": "ROW", "id": f"ROW-{i}",
                            "children": [g["id"] for g in grid[i*2:i*2+2]],
                            "parents": ["GRID_ID"]}
               for i in range((len(grid)+1)//2)},
            **{g["id"]: {**g, "parents": [f"ROW-{i//2}"]} for i, g in enumerate(grid)},
        }),
    }

# ------------------------------------------------------------------
# 4. — orchestration
# ------------------------------------------------------------------
def run():
    # 4a. DB views
    apply_views()

    # 4b. Superset
    client = SupersetClient(SUPERSET_URL, SUPERSET_USER, SUPERSET_PASSWORD)
    db_id = client.ensure_database(SUPERSET_DB_NAME, POSTGRES_URL)

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