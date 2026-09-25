# build_app.spec
# ----------------
# PyInstaller spec that packages desktop_app into a single standalone
# executable (no Python install, no terminal, no venv needed to run it).
#
# Build it (on Windows, inside the activated venv, from the project root):
#     pip install pyinstaller
#     pyinstaller build_app.spec
#
# The finished .exe appears at: dist/SurveillanceApp/SurveillanceApp.exe
# Zip up the whole dist/SurveillanceApp/ folder to share it -- it's
# self-contained (includes its own Python + all dependencies).
#
# First build will take a few minutes and produces a large folder (RT-DETR's
# dependencies -- torch, ultralytics -- are not small). That's normal.

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# ultralytics and torch both need their non-code data files (model configs,
# etc.) pulled in explicitly -- PyInstaller can't always find these on its own.
for pkg in ("ultralytics", "torch", "torchvision"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# bundle the default zone config and cameras.yaml example alongside the app
datas += [("config", "config")]

a = Analysis(
    ["desktop_app/main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SurveillanceApp",
    debug=False,
    strip=False,
    upx=True,
    console=False,  # no terminal window -- double-click launches the GUI directly
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="SurveillanceApp",
)
