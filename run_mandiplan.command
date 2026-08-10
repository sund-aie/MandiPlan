#!/bin/bash
# Double-click this file in Finder to run MandiPlan.
#
# On first run it creates a private virtual environment next to this script and
# installs the dependencies into it; afterwards it just starts the application.
# It never touches your system Python packages.

set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "MandiPlan needs Python 3.11 or newer."
    echo "Install it from https://www.python.org/downloads/ and run this again."
    echo
    read -r -p "Press return to close."
    exit 1
fi

VERSION=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$VERSION" in
    3.1[1-9]|3.[2-9][0-9]) ;;
    *)
        echo "MandiPlan needs Python 3.11 or newer; python3 here is $VERSION."
        read -r -p "Press return to close."
        exit 1
        ;;
esac

if [ ! -x .venv/bin/python ]; then
    echo "First run: setting up a virtual environment in $(pwd)/.venv"
    python3 -m venv .venv
    .venv/bin/python -m pip install --quiet --upgrade pip
    echo "Installing dependencies (this downloads about 150 MB, once)…"
    .venv/bin/python -m pip install --quiet -r requirements.txt
    echo "Done."
fi

exec .venv/bin/python -m mandiplan "$@"
