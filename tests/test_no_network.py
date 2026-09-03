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


def test_dataset_fetching_lives_outside_the_application():
    """Downloading is a deliberate terminal step, not something the app does.

    tools/ is exempt from the no-network rule above precisely because it is not
    imported by the application; this keeps that exemption honest.
    """
    fetcher = PACKAGE.parent / "tools" / "fetch_datasets.py"
    assert fetcher.exists()
    assert any(name.startswith("urllib") for name in _imported_modules(fetcher))
    for path in PACKAGE.rglob("*.py"):
        imported = _imported_modules(path)
        assert not any("fetch_datasets" in name for name in imported), path


def test_the_status_bar_carries_no_disclaimer_banner():
    """The status-bar banner was removed at the project owner's instruction.

    The disclaimer still reaches the About dialog and every exported CSV; it is
    simply no longer painted across the bottom of the window.
    """
    source = (PACKAGE / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "self.banner" not in source
