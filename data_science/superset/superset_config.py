import os

SECRET_KEY = os.getenv("SUPERSET_SECRET_KEY")

# SQLite metadata DB by default
SQLALCHEMY_DATABASE_URI = os.getenv(
    "SUPERSET_DATABASE_URI", 
    "sqlite:////app/superset_home/superset.db"
)

# Feature flags
FEATURE_FLAGS = {
    "DASHBOARD_RBAC": True,
    "ENABLE_TEMPLATE_PROCESSING": True,
}

# Session / security
SESSION_COOKIE_SECURE = False         # set True behind HTTPS
WTF_CSRF_ENABLED = True               # keep True; API uses CSRF token anyway