"""Extract and optionally verify the complete dataset shipped with the source repository."""

import argparse
import json
import sys
from pathlib import Path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Compare every extracted file.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from prognostics.dataset_bundle import prepare_bundled_data, verify_bundled_data

    try:
        data = prepare_bundled_data(root)
        if data is None:
            raise ValueError("No bundled archive found. Use the GitHub source checkout.")
        print("Dataset ready:", data)
        if args.verify:
            print(json.dumps(verify_bundled_data(root), indent=2))
    except (ValueError, OSError) as exc:
        print("Unable to prepare dataset:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
