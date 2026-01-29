"""Convenience module to run CLI via `python -m pipeline`."""

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> int:
    _ensure_src_on_path()
    from pipeline.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
