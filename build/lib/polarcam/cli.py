from __future__ import annotations
"""
PolarCam launcher.

- Creates the QApplication.
- Installs a Qt message handler -> Python logging bridge.
- Sets up a user-friendly exception hook (shows a dialog & logs the traceback).
- Keeps terminal mostly quiet (only warnings/errors) unless -v flags are used.
"""

import argparse
import sys
import traceback
from typing import Optional

from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QMessageBox

from polarcam import (
    __app_name__,
    __version__,
    setup_logging,
    install_qt_message_handler,
)
from polarcam.controller.controller import Controller
from polarcam.app.main_window import MainWindow


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=__app_name__, add_help=True)
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase logging verbosity (-v: INFO, -vv: DEBUG)."
    )
    p.add_argument(
        "--log-file", default=None,
        help="Optional path to write a log file (rotating)."
    )
    return p.parse_args(argv)


def _install_exception_hook(logger):
    """Show a dialog on uncaught exceptions and log the traceback."""
    def _hook(exc_type, exc, tb):
        logger.error("Uncaught exception", exc_info=(exc_type, exc, tb))
        try:
            text = "".join(traceback.format_exception_only(exc_type, exc)).strip()
            QMessageBox.critical(
                None,
                f"{__app_name__} — Unhandled error",
                f"{text}\n\n(See terminal or log for details.)"
            )
        except Exception:
            traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
    sys.excepthook = _hook


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Configure logging first, then bridge Qt messages into it.
    logger = setup_logging(verbosity=args.verbose, logfile=args.log_file)

    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("BFM Lab")
    app.setOrganizationDomain("bacteria.motors.local")

    # Bridge Qt warnings/errors -> Python logging
    qInstallMessageHandler(install_qt_message_handler(logger))

    # Friendlier crash handling
    _install_exception_hook(logger)

    # Build + show main window
    ctrl = Controller()
    win = MainWindow(ctrl)
    win.show()

    # Ensure a single shutdown path for camera resources
    app.aboutToQuit.connect(win.safe_shutdown)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
