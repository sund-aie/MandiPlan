"""Entry point: ``python -m mandiplan``."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from . import safety
from .constants import APP_NAME


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    # Before anything else: an exception in a click handler must not abort the
    # process and take the plan with it.
    safety.install()
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)

    from .ui.theme import stylesheet

    app.setStyleSheet(stylesheet())

    from .ui.main_window import MainWindow

    window = MainWindow()
    safety.set_reporter(window.show_error)
    window.show()
    # A plain Python process does not come to the front on its own, so the
    # window can otherwise open behind the terminal it was launched from.
    window.raise_()
    window.activateWindow()
    window.start()
    if len(argv) > 1:
        window.load_folder(argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
