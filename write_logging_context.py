from pathlib import Path

content = '''"""Context propagation helpers for AgentForge logging and request correlation.

This module exposes lightweight ``contextvars``-based helpers that let
correlation IDs, session IDs, and task IDs flow through nested operations
without passing them explicitly through every function call.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator

__all__ = [
    "correlation_scope",
    "get_correlation_id",
    "get_session_id",
    "get_task_id",
    "new_correlation_id",
    "session_scope",
    "task_scope",
]

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)
_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "task_id", default=None
)


def new_correlation_id() -> str:
    """Create a new correlation ID for a request or task."""
    import uuid

    return str(uuid.uuid4())


def get_correlation_id() -> str | None:
    """Return the active correlation ID, if any."""
    return _correlation_id.get()


def get_session_id() -> str | None:
    """Return the active session ID, if any."""
    return _session_id.get()


def get_task_id() -> str | None:
    """Return the active task ID, if any."""
    return _task_id.get()


def correlation_scope(correlation_id: str) -> Iterator[None]:
    """Temporarily bind a correlation ID for the current context."""
    token = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.reset(token)


def session_scope(session_id: str) -> Iterator[None]:
    """Temporarily bind a session ID for the current context."""
    token = _session_id.set(session_id)
    try:
        yield
    finally:
        _session_id.reset(token)


def task_scope(task_id: str) -> Iterator[None]:
    """Temporarily bind a task ID for the current context."""
    token = _task_id.set(task_id)
    try:
        yield
    finally:
        _task_id.reset(token)
'''

Path('src/agentforge/logging/context.py').write_text(content, encoding='utf-8')
print('wrote logging context')
