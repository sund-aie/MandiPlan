"""Fetch and register the open CBCT datasets that back the shape library.

This script lives outside the ``mandiplan`` package on purpose: the
application makes no network calls, and tests/test_no_network.py enforces
that. Fetching is something you run once, deliberately, from a terminal.

    python3 tools/fetch_datasets.py --list
    python3 tools/fetch_datasets.py --dataset margo
    python3 tools/fetch_datasets.py --all --data-root data/cbct

Each dataset lands in ``data/cbct/<name>/`` with a manifest.json recording the
source, the licence as published by the repository, and the case count.

Two guards run before anything is written:

* the licence returned by the repository must be on the allowlist, and must
  match what the registry claims. A CC BY-NC-ND dataset, or anything from a
  TCIA collection, is refused here rather than left for you to remember.
* the registry's own metadata is treated as unverified. Where the repository
  publishes a licence, that is what is checked and recorded, not the claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mandiplan.cohort import licence_is_allowed, load_registry, source_is_allowed

ZENODO_API = "https://zenodo.org/api/records/{record}"
FIGSHARE_API = "https://api.figshare.com/v2/articles/{record}"
TIMEOUT = 60


class IngestionRefused(RuntimeError):
    """A dataset failed a licence or source check and was not downloaded."""


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def describe_record(record_id: str) -> dict:
    """Ask the repository what a record actually is, licence included."""
    host, _, number = record_id.partition(":")
    if host == "zenodo":
        payload = _get_json(ZENODO_API.format(record=number))
        metadata = payload.get("metadata", {})
        licence = metadata.get("license", {})
        return {
            "source": f"https://zenodo.org/records/{number}",
            "title": metadata.get("title", ""),
            "licence": licence.get("id") or licence.get("identifier") or "",
            "files": [
                {"name": f.get("key"), "size": f.get("size"), "url": f["links"]["self"]}
                for f in payload.get("files", [])
                if f.get("links", {}).get("self")
            ],
        }
    if host == "figshare":
        payload = _get_json(FIGSHARE_API.format(record=number))
        return {
            "source": f"https://figshare.com/articles/{number}",
            "title": payload.get("title", ""),
            "licence": (payload.get("license") or {}).get("name", ""),
            "files": [
                {"name": f.get("name"), "size": f.get("size"), "url": f.get("download_url")}
                for f in payload.get("files", [])
                if f.get("download_url")
            ],
        }
    raise IngestionRefused(f"unknown repository in record id {record_id!r}")


def _normalise_licence(text: str) -> str:
    """Map the many spellings a repository uses onto the registry's ids."""
    lowered = (text or "").lower().replace("_", "-").replace(" ", "-")
    if "nc" in lowered.split("-") or "noncommercial" in lowered:
        return "CC-BY-NC-4.0" if "nd" not in lowered.split("-") else "CC-BY-NC-ND-4.0"
    if "nd" in lowered.split("-") or "noderivatives" in lowered:
        return "CC-BY-ND-4.0"
    if lowered.startswith("cc0") or "zero" in lowered:
        return "CC0-1.0"
    if "sa" in lowered.split("-") or "sharealike" in lowered:
        return "CC-BY-SA-4.0"
    if lowered.startswith("cc-by") or "attribution" in lowered:
        return "CC-BY-4.0"
    return text or ""


def check_allowed(entry: dict, published_licence: str, source: str, registry: dict) -> str:
    """Refuse anything the policy excludes. Returns the licence to record."""
    allowed, reason = source_is_allowed(source, registry)
    if not allowed:
        raise IngestionRefused(reason)

    licence = _normalise_licence(published_licence) or entry["licence"]
    allowed, reason = licence_is_allowed(licence, registry)
    if not allowed:
        raise IngestionRefused(reason)

    claimed = entry["licence"]
    if published_licence and licence != claimed:
        raise IngestionRefused(
            f"the registry claims {claimed} but the repository publishes "
            f"{published_licence!r} ({licence}); refusing until the registry is corrected"
        )
    return licence


def download(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    return destination.stat().st_size


def write_manifest(folder: Path, entry: dict, described: list[dict], licence: str,
                   case_count: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": entry["name"],
                "title": entry["title"],
                "kind": entry["kind"],
                "pool": entry.get("pool", "primary"),
                "licence": licence,
                "licence_status": entry.get("licence_status", ""),
                "case_count": case_count,
                "sources": [d["source"] for d in described],
                "records": entry.get("records", []),
                "requires_truncation_filter": entry.get("requires_truncation_filter", False),
                "demographics": entry.get("demographics", {}),
                "fetched_on": date.today().isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def fetch(entry: dict, data_root: Path, registry: dict, dry_run: bool) -> None:
    folder = data_root / entry["name"]
    if not entry.get("records"):
        raise IngestionRefused(
            "no repository record id is registered for this dataset; add one to "
            "mandiplan/data/datasets.json before it can be fetched"
        )

    described = []
    licence = entry["licence"]
    for record_id in entry["records"]:
        info = describe_record(record_id)
        licence = check_allowed(entry, info["licence"], info["source"], registry)
        described.append(info)

    total_files = sum(len(info["files"]) for info in described)
    print(f"  licence {licence} accepted; {total_files} file(s) available")
    if dry_run:
        print("  dry run: nothing downloaded")
        return

    count = 0
    for info in described:
        for item in info["files"]:
            target = folder / "files" / item["name"]
            if target.exists():
                continue
            size = download(item["url"], target)
            count += 1
            print(f"  {item['name']} ({size / 1e6:.1f} MB)")
    write_manifest(folder, entry, described, licence, entry.get("case_count") or count)
    print(f"  manifest written to {folder / 'manifest.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data-root", default="data/cbct")
    parser.add_argument("--dry-run", action="store_true", help="check licences only")
    args = parser.parse_args(argv)

    registry = load_registry()
    if args.list:
        for entry in registry["datasets"]:
            flag = "" if entry.get("metadata_verified") else "  [metadata unverified]"
            print(f"{entry['name']:26s} {entry['licence']:14s} {entry['title']}{flag}")
        print("\nHard-excluded:")
        for item in registry["hard_excluded"]:
            print(f"  {item.get('name', item.get('pattern'))}: {item['reason']}")
        return 0

    wanted = registry["datasets"] if args.all else [
        entry for entry in registry["datasets"] if entry["name"] in args.dataset
    ]
    if not wanted:
        parser.error("choose --all, or --dataset NAME (see --list)")

    failures = 0
    for entry in wanted:
        print(f"{entry['name']}:")
        try:
            fetch(entry, Path(args.data_root), registry, args.dry_run)
        except IngestionRefused as error:
            failures += 1
            print(f"  REFUSED: {error}")
        except (urllib.error.URLError, OSError) as error:
            failures += 1
            print(f"  could not reach the repository: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
