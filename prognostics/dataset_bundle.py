"""Prepare the versioned source dataset without changing existing local data."""

import hashlib
import json
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

DATASET_NAME = "Multimodal data for prognostics"


def bundle_paths(workspace):
    workspace = Path(workspace)
    return (
        workspace / "data/raw" / f"{DATASET_NAME}.zip",
        workspace / "data/dataset-manifest.json",
        workspace / "data/raw" / DATASET_NAME,
    )


def checked_archive(archive, manifest):
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("Bundled dataset checksum mismatch; download the archive again.")
    return metadata


def checked_members(zipped, metadata):
    files, seen = [], set()
    for item in zipped.infolist():
        path = PurePosixPath(item.filename)
        if (
            path.is_absolute()
            or "\\" in item.filename
            or any(part in {".", ".."} or ":" in part for part in path.parts)
            or not path.parts
            or path.parts[0] != DATASET_NAME
            or stat.S_ISLNK(item.external_attr >> 16)
            or item.filename in seen
        ):
            raise ValueError("Unsafe or duplicate path in bundled dataset.")
        seen.add(item.filename)
        if not item.is_dir():
            files.append(item)
    if (
        len(files) != metadata["file_count"]
        or sum(p.file_size for p in files) != metadata["uncompressed_bytes"]
    ):
        raise ValueError("Bundled dataset file count or size mismatch.")
    return files


def prepare_bundled_data(workspace):
    archive, manifest, destination = bundle_paths(workspace)
    if (destination / "1/time.txt").is_file():
        return destination
    if not archive.is_file() or not manifest.is_file():
        return None
    if destination.exists():
        raise ValueError("Dataset directory exists but is incomplete; move it before preparation.")
    metadata = checked_archive(archive, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            ZipFile(archive) as zipped,
            tempfile.TemporaryDirectory(prefix=".dataset-", dir=destination.parent) as staging,
        ):
            checked_members(zipped, metadata)
            zipped.extractall(staging)
            prepared = Path(staging) / DATASET_NAME
            if not (prepared / "1/time.txt").is_file():
                raise ValueError("Bundled dataset has no 1/time.txt.")
            prepared.rename(destination)
    except BadZipFile as exc:
        raise ValueError("Bundled dataset is corrupt; download the archive again.") from exc
    return destination


def verify_bundled_data(workspace):
    archive, manifest, destination = bundle_paths(workspace)
    metadata = checked_archive(archive, manifest)
    try:
        with ZipFile(archive) as zipped:
            files = checked_members(zipped, metadata)
            for item in files:
                relative = PurePosixPath(item.filename).relative_to(DATASET_NAME)
                path = destination.joinpath(*relative.parts)
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).digest()
                    != hashlib.sha256(zipped.read(item)).digest()
                ):
                    raise ValueError(f"Extracted dataset differs from the archive: {relative}")
    except BadZipFile as exc:
        raise ValueError("Bundled dataset is corrupt.") from exc
    return {
        "files_verified": len(files),
        "uncompressed_bytes": metadata["uncompressed_bytes"],
        "archive_sha256": metadata["archive_sha256"],
    }
