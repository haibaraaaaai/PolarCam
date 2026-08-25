"""
polarcam package bootstrap.

Holds:
- App metadata (name/version)
- Logging setup helper (console quiet by default; optional rotating file)
- Qt message handler factory that routes Qt warnings/errors to logging
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Callable, Optional

__app_name__ = "PolarCam"
__version__ = "0.1.0"


def setup_logging(*, verbosity: int = 0, logfile: Optional[str] = None) -> logging.Logger:
    """
    Configure root logger for the app.
    - verbosity: 0=WARNING (default), 1=INFO, >=2=DEBUG
    - logfile: optional path to a rotating log file
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logger = logging.getLogger(__app_name__.lower())
    logger.setLevel(level)

    # Clear duplicated handlers if we re-enter (e.g., tests / hot reloads)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler — quiet by default
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional rotating file handler (keeps more detail even when console is quiet)
    if logfile:
        log_path = Path(logfile)
    else:
        # Default under user home: ~/.polarcam/polarcam.log
        base = Path(os.path.expanduser("~")) / ".polarcam"
        base.mkdir(parents=True, exist_ok=True)
        log_path = base / "polarcam.log"

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)  # always capture full detail to the file
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Reduce noisy third-party loggers unless explicitly requested
    for noisy in ("asyncio", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    logger.debug("Logging initialized (level=%s, file=%s)", logging.getLevelName(level), log_path)
    return logger


# ---- Qt message handler bridge ---------------------------------------------

def install_qt_message_handler(logger: logging.Logger) -> Callable:
    """
    Returns a Qt message handler function that forwards Qt messages to logging.

    Usage:
        from PySide6.QtCore import qInstallMessageHandler
        qInstallMessageHandler(install_qt_message_handler(logger))
    """
    from PySide6.QtCore import QtMsgType, QLoggingCategory  # imported lazily

    # Optionally quiet very chatty Qt categories here:
    QLoggingCategory.setFilterRules(
        """
        *.debug=false
        qt.qpa.input.debug=false
        """
    )

    def handler(mode, context, message):
        # Map Qt message types to logging levels
        if mode == QtMsgType.QtDebugMsg:
            lvl = logging.DEBUG
        elif mode == QtMsgType.QtInfoMsg:
            lvl = logging.INFO
        elif mode == QtMsgType.QtWarningMsg:
            lvl = logging.WARNING
        elif mode == QtMsgType.QtCriticalMsg:
            lvl = logging.ERROR
        elif mode == QtMsgType.QtFatalMsg:
            lvl = logging.CRITICAL
        else:
            lvl = logging.WARNING

        # Compose a compact origin string
        if context and context.function:
            origin = f"{context.file}:{context.line} in {context.function}"
        elif context and context.file:
            origin = f"{context.file}:{context.line}"
        else:
            origin = "qt"

        logger.log(lvl, "[Qt] %s — %s", origin, message)

        # For fatal messages, Qt wants us to abort — keep the default behavior.
        if mode == QtMsgType.QtFatalMsg:
            import sys
            sys.stderr.write(message + "\n")
            sys.stderr.flush()

    return handler
