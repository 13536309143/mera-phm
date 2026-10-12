"""Check tracked release files; never print credential matches or scan private artifacts."""

import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = [
    re.compile(rb"sk-(?:cp-)?[A-Za-z0-9_-]{24,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"/" + rb"Users" + rb"/[^/\s]+/"),
    re.compile(rb"/" + rb"Volumes" + rb"/OUT/"),
]
FORBIDDEN = [".env", ".venv/", "artifacts/", "runs/", "data/raw/", "data/demo/"]


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    names = [n.decode() for n in result.stdout.split(b"\0") if n]
    failures = []
    for name in names:
        path = ROOT / name
        if (
            name == ".env"
            or any(name.startswith(prefix) for prefix in FORBIDDEN[1:])
            or path.suffix == ".joblib"
        ):
            failures.append((name, "private/generated file tracked"))
            continue
        blobs = [path.read_bytes()]
        if path.suffix == ".zip":
            with ZipFile(path) as archive:
                blobs = [archive.read(n) for n in archive.namelist() if not n.endswith("/")]
        if any(pattern.search(blob) for pattern in PATTERNS for blob in blobs):
            failures.append((name, "credential-like content or private absolute path"))
    if failures:
        for name, reason in failures:
            print(name + ": " + reason)
        return 1
    print(
        f"Release scan passed: {len(names)} tracked files; no real keys or private paths detected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
