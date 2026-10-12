"""Portable launcher shared by the macOS and Windows entry points."""

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def interpreter(root=ROOT, windows=None):
    windows = os.name == "nt" if windows is None else windows
    return Path(root) / ".venv" / ("Scripts/python.exe" if windows else "bin/python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["serve", "local", "api"], default="serve")
    args, rest = parser.parse_known_args()
    if sys.version_info < (3, 12):
        print("MERA-PHM requires Python 3.12 or later.")
        return 1
    python = interpreter()
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    check = subprocess.run(
        [
            str(python),
            "-c",
            "from importlib.metadata import version; assert version('mera-phm') == '1.1.0'",
        ],
        cwd=ROOT,
        capture_output=True,
    )
    if check.returncode:
        subprocess.run([str(python), "-m", "pip", "install", "-e", str(ROOT)], check=True)
    command = [str(python), str(ROOT / "main.py")]
    has_model = (ROOT / "artifacts/main-model/model-bank.joblib").is_file()
    explicit = any(x == "--model" or x.startswith("--model=") for x in rest)
    extra = []
    if not has_model and not explicit:
        if not (ROOT / "artifacts/demo-model/model-bank.joblib").is_file():
            demo = command + ["demo"]
            if (ROOT / "data/demo/1/time.txt").is_file():
                demo = command + [
                    "train",
                    "--data",
                    "data/demo",
                    "--output",
                    "artifacts/demo-model",
                ]
            subprocess.run(demo, cwd=ROOT, check=True)
        extra = ["--data", "data/demo", "--model", "artifacts/demo-model/model-bank.joblib"]
    if args.mode == "serve":
        final = command + ["serve", "--open", *extra, *rest]
    else:
        final = command + ["predict", "--mode", args.mode, "--row", "8", *extra, *rest]
    return subprocess.call(final, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
