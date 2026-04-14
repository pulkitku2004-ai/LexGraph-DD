"""
Neo4j driver singleton.

Neo4j's official Python driver manages a connection pool internally.
One Driver instance per process is the intended usage pattern — the driver
is thread-safe and session creation is cheap. Creating a new Driver per
request would exhaust the bolt port pool under load.

Why bolt:// not neo4j://?
- bolt:// connects to a single instance (what we're running locally and in Docker).
- neo4j:// is for Aura / cluster routing. We'll add a settings flag to switch
  when moving to production.

Session management: always use driver.session() as a context manager.
The driver does NOT auto-close sessions — unclosed sessions leak bolt connections.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from neo4j import Driver, GraphDatabase, Session

from core.config import settings

_driver: Driver | None = None


def get_neo4j_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


@contextmanager
def get_neo4j_session() -> Generator[Session, None, None]:
    """
    Context manager that guarantees session closure.
    Usage:
        with get_neo4j_session() as session:
            session.run("MATCH (n) RETURN n LIMIT 1")
    """
    driver = get_neo4j_driver()
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def check_neo4j_health() -> bool:
    """
    Returns True if Neo4j is reachable and can execute a trivial query.
    Does NOT raise.
    """
    try:
        with get_neo4j_session() as session:
            session.run("RETURN 1")
        return True
    except Exception:
        return False


def close_neo4j_driver() -> None:
    """Call this on application shutdown to drain the connection pool cleanly."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
