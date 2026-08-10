#!/bin/bash
# Build MandiPlan.app (and a .dmg to hand around) on macOS.
#
#   ./packaging/build_macos.sh
#
# The result is dist/MandiPlan.app and dist/MandiPlan.dmg. The app is not code
# signed or notarised, so the first launch needs a right-click -> Open, or
# System Settings -> Privacy & Security -> Open Anyway.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

if [ "$(uname)" != "Darwin" ]; then
    echo "This script builds the macOS app bundle; run it on macOS."
    exit 1
fi

echo "==> Python environment"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt pyinstaller

echo "==> Icon"
.venv/bin/python packaging/make_icon.py
ICONSET="$ROOT/packaging/MandiPlan.iconset"
rm -rf "$ICONSET" "$ROOT/packaging/MandiPlan.icns"
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z $size $size packaging/icon.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double packaging/icon.png \
        --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$ROOT/packaging/MandiPlan.icns"
rm -rf "$ICONSET"

echo "==> Bundle"
rm -rf build/pyinstaller dist/MandiPlan.app dist/MandiPlan dist/MandiPlan.dmg
.venv/bin/python -m PyInstaller \
    --noconfirm --clean \
    --distpath dist --workpath build/pyinstaller \
    packaging/mandiplan.spec

if [ ! -d dist/MandiPlan.app ]; then
    echo "PyInstaller did not produce dist/MandiPlan.app"
    exit 1
fi

echo "==> Disk image"
STAGING=$(mktemp -d)
cp -R dist/MandiPlan.app "$STAGING/"
ln -s /Applications "$STAGING/Applications"
hdiutil create -volname "MandiPlan" -srcfolder "$STAGING" -ov -format UDZO \
    dist/MandiPlan.dmg >/dev/null
rm -rf "$STAGING"

echo
echo "Built:"
echo "  dist/MandiPlan.app"
echo "  dist/MandiPlan.dmg   (drag MandiPlan onto Applications to install)"
echo
echo "The bundle is unsigned. First launch: right-click the app -> Open."
