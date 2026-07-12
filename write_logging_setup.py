from pathlib import Path

content = '''"""Structured logging configuration for AgentForge, built on ``structlog``.

Configures a single, process-wide ``structlog`` pipeline that:

* renders either JSON (for production/API/telemetry ingestion) or
  human-readable console output (for local CLI use), controlled by
  :class:`agentforge.config.settings.LoggingSettings`,
* automatically injects the active correlation/session/task IDs from
  :mod:`agentforge.logging.context` into every log event,
* routes standard library ``logging`` calls (from third-party
  dependencies) through the same pipeline for consistent output.

Every module in AgentForge obtains its logger via :func:`get_logger`
rather than calling ``structlog.get_logger`` directly, so that the import
surface for "how do I log in this codebase" is a single, documented
function.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from agentforge.config.settings import LoggingSettings
from agentforge.logging.context import get_correlation_id, get_session_id, get_task_id

__all__ = ["configure_logging", "get_logger"]

_configured: bool = False


def _inject_context_ids(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    correlation_id = get_correlation_id()
    if correlation_id is not None:
        event_dict["correlation_id"] = correlation_id
    session_id = get_session_id()
    if session_id is not None:
        event_dict["session_id"] = session_id
    task_id = get_task_id()
    if task_id is not None:
        event_dict["task_id"] = task_id
    return event_dict


def configure_logging(settings: LoggingSettings) -> None:
    global _configured

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context_ids,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=logging.getLevelName(settings.level),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
'''

Path('src/agentforge/logging/setup.py').write_text(content, encoding='utf-8')
print('wrote logging setup')
