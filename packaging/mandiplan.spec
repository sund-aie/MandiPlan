# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of MandiPlan.

Used by packaging/build_macos.sh; also builds on Linux, which is how the
collected imports below are checked.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected

datas = [(str(ROOT / "mandiplan" / "data"), "mandiplan/data")]
binaries = []
hiddenimports = [
    "vtkmodules.all",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "SimpleITK",
    "pydicom",
]

# VTK ships as many separate extension modules that are only reachable through
# vtkmodules.all, so it has to be collected wholesale.
vtk_datas, vtk_binaries, vtk_hidden = collect_all("vtkmodules")
datas += vtk_datas
binaries += vtk_binaries
hiddenimports += vtk_hidden

# SimpleITK and pydicom only need their data files and libraries. Collecting
# their submodules as well would import pydicom's optional encryption and
# pixel-handler backends, which are not used here and fail to import on
# machines where the optional dependency is broken.
for package in ("SimpleITK", "pydicom"):
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)

analysis = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # MandiPlan makes no network calls and does not read encrypted DICOM, so
    # the TLS and plotting stacks are dead weight in the bundle.
    excludes=[
        "tkinter",
        "matplotlib",
        "PyQt5",
        "PySide6",
        "cryptography",
        "requests",
        "urllib3",
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure)  # noqa: F821

executable = EXE(  # noqa: F821
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="MandiPlan",
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / "packaging" / "MandiPlan.icns") if sys.platform == "darwin" else None,
)

collection = COLLECT(  # noqa: F821
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="MandiPlan",
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        collection,
        name="MandiPlan.app",
        icon=str(ROOT / "packaging" / "MandiPlan.icns"),
        bundle_identifier="org.mandiplan.app",
        version="0.1.0",
        info_plist={
            "CFBundleName": "MandiPlan",
            "CFBundleDisplayName": "MandiPlan",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": (
                "RESEARCH AND EDUCATION USE ONLY - NOT A MEDICAL DEVICE"
            ),
        },
    )
