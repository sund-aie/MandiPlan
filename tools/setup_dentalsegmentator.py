"""Point MandiPlan at a DentalSegmentator install. Nothing is vendored.

The nnU-Net weights are large and separately licensed, so they are installed
by you and MandiPlan only records where they are:

    python3 tools/setup_dentalsegmentator.py --path ~/models/DentalSegmentator
    python3 tools/setup_dentalsegmentator.py --show

The path is written to config.json in the data root, and the application reads
it from there. Nothing is downloaded by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mandiplan.cohort import load_registry


def config_path(data_root: Path) -> Path:
    return data_root / "config.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", help="local directory holding the model weights")
    parser.add_argument("--data-root", default="data/cbct")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    target = config_path(root)
    config = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}

    model = load_registry()["external_models"][0]
    if args.show or not args.path:
        print(f"model            : {model['title']}")
        print(f"published at     : {', '.join(model['records'])}")
        print(f"licence (claimed): {model['licence']}  [metadata unverified]")
        print(f"configured path  : {config.get('dentalsegmentator_path') or '(not set)'}")
        print("labels           :")
        for value, name in sorted(model["labels"].items()):
            print(f"  {value} = {name}")
        return 0

    install = Path(args.path).expanduser()
    if not install.is_dir():
        print(f"{install} is not a directory; install the weights there first.")
        return 1
    config["dentalsegmentator_path"] = str(install.resolve())
    config["dentalsegmentator_labels"] = model["labels"]
    root.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {install.resolve()} in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
