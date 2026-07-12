from pathlib import Path

content = '''"""Structured logging for AgentForge.

This package provides:

* :mod:`agentforge.logging.setup` — process-wide ``structlog`` pipeline
  configuration (:func:`~agentforge.logging.setup.configure_logging`) and
  the canonical logger factory (:func:`~agentforge.logging.setup.get_logger`).
* :mod:`agentforge.logging.context` — ``contextvars``-based correlation ID,
  session ID, and task ID propagation, automatically merged into every log
  event by the setup pipeline.

Usage convention: every module that needs to log calls
``get_logger(__name__)`` once at module level and reuses that logger
instance for all logging calls within the module, e.g.::

    from agentforge.logging import get_logger

    logger = get_logger(__name__)

    def do_something() -> None:
        logger.info("doing_something", detail="value")
"""

from __future__ import annotations

from agentforge.logging.context import (
    correlation_scope,
    get_correlation_id,
    get_session_id,
    get_task_id,
    new_correlation_id,
    session_scope,
    task_scope,
)
from agentforge.logging.setup import configure_logging, get_logger

__all__ = [
    "configure_logging",
    "get_logger",
    "correlation_scope",
    "get_correlation_id",
    "get_session_id",
    "get_task_id",
    "new_correlation_id",
    "session_scope",
    "task_scope",
]
'''

Path('src/agentforge/logging/__init__.py').write_text(content, encoding='utf-8')
print('wrote logging init')
