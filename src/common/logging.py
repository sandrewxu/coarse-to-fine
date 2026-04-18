"""Project-wide logging.

A single configured logger named ``c2f``; modules pull a child via
``get_logger(__name__)``. Configuration honours the ``LOG_LEVEL`` env var
(default ``INFO``), routes to stderr, and uses a compact format with
millisecond timestamps. The logger is configured exactly once on first call,
so import order doesn't matter.
"""

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("c2f")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return the project logger for ``name``, configuring on first call.

    Strips a leading ``src.`` prefix so log records read ``c2f.rl.reward``
    instead of ``c2f.src.rl.reward``.
    """
    _configure()
    short = name.removeprefix("src.")
    return logging.getLogger(f"c2f.{short}")
