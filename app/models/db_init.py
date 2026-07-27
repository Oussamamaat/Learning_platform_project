"""
Database Schema and Vector Extension Initialization
───────────────────────────────────────────────────
Enables pgvector and initializes tables in PostgreSQL.

Usage:
    python -m app.models.db_init
"""
import sys
from sqlalchemy import create_engine, text
from app.config import get_settings
from app.models.database import Base


def init_db():
    settings = get_settings()
    db_url = settings.database_url
    # Print database host for confirmation, hiding credentials
    safe_db_url = db_url.split("@")[-1] if "@" in db_url else db_url
    print(f"Connecting to database to initialize: {safe_db_url}")

    try:
        # Create engine
        engine = create_engine(db_url)

        # 1. Enable pgvector extension
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            print("Enabling pgvector extension...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # 2. Create tables
        print("Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("Database tables initialized successfully.")

    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    init_db()
