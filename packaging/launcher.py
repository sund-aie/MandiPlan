"""Frozen-application entry point (PyInstaller runs this, not ``__main__``)."""

import sys

from mandiplan.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
