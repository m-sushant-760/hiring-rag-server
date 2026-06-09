"""
Database utility — manages the PostgreSQL connection pool lazily.
This prevents connection attempts at import time, ensuring that unit tests
can collect and execute without requiring a running database.
"""

from psycopg_pool import ConnectionPool
from src.config import settings

_pool = None

def get_pool() -> ConnectionPool:
    """Lazily initializes and returns the global connection pool."""
    global _pool
    if _pool is None:
        db_url = settings.database_url
        if not db_url:
            raise ValueError(
                "DATABASE_URL setting is empty or not set! "
                "Ensure it is configured in your environment or .env file."
            )
        _pool = ConnectionPool(
            conninfo=db_url,
            min_size=1,
            max_size=10,
            open=True
        )
    return _pool

def close_pool() -> None:
    """Closes the connection pool and cleans up active connections."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
