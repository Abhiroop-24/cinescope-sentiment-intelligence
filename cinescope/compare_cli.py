from __future__ import annotations

import argparse
from pathlib import Path

from .compare import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the six-model IMDb comparison")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/model_comparison.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(run(args.data_root, args.output, args.limit))


if __name__ == "__main__":
    main()
