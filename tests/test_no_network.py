"""MandiPlan never transmits anything. This checks the code cannot."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "mandiplan"

FORBIDDEN = {
    "socket",
    "ssl",
    "http",
    "http.client",
    "urllib",
    "urllib.request",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "requests",
    "httpx",
    "aiohttp",
    "boto3",
    "webbrowser",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_no_module_imports_anything_that_can_reach_the_network():
    offenders = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        bad = {
            name
            for name in _imported_modules(path)
            if name in FORBIDDEN or name.split(".")[0] in FORBIDDEN
        }
        if bad:
            offenders[str(path.relative_to(PACKAGE.parent))] = sorted(bad)
    assert not offenders, f"network-capable imports found: {offenders}"


def test_the_disclaimer_is_not_configurable():
    from mandiplan.constants import DISCLAIMER

    assert "NOT A MEDICAL DEVICE" in DISCLAIMER
    source = (PACKAGE / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "addPermanentWidget(self.banner)" in source
