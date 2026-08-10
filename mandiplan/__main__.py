"""Entry point: ``python -m mandiplan``."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .constants import APP_NAME


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)

    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    window.start()
    if len(argv) > 1:
        window.load_folder(argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
