"""Run native TubeletGraph inference from its checkout, without preview rendering."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-mask", type=Path, required=True)
    parser.add_argument("--fps", type=int, required=True)
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Input frame directory does not exist: {args.input_dir}")
    if not args.input_mask.is_file():
        parser.error(f"Input mask does not exist: {args.input_mask}")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    # This helper lives outside the checkout; upstream imports are relative to it.
    sys.path.insert(0, str(Path.cwd()))
    create_custom_dataset = import_module("eval.build_custom_dataset").create_custom_dataset
    config_path, video_name = create_custom_dataset(
        "configs/default.yaml", str(args.input_dir), str(args.input_mask), fps=args.fps
    )
    subprocess.run(
        [sys.executable, "TubeletGraph/run.py", "-c", str(config_path),
         "-d", "custom", "-s", video_name, "-m", "Ours"],
        check=True,
    )


if __name__ == "__main__":
    main()
