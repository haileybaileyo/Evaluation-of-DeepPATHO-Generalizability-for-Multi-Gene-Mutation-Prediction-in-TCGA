"""Copy the SMALL result files of every experiment from runs/ (git-ignored) into
results/ (tracked), so the numbers behind RESULTS.md can be inspected without
re-running anything.

Rules: only .json / .csv files below SIZE_LIMIT; never checkpoints (*.pt),
logs, or per-patch score tables (they are large and re-creatable).

Usage:
    python scripts/collect_results.py            # copy
    python scripts/collect_results.py --dry-run  # only list what would be copied
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE_LIMIT = 2 * 1024 * 1024
KEEP_SUFFIX = {".json", ".csv"}
SKIP_NAME_PARTS = ("patch_scores", "resume_checkpoint", "selected_patches", "label_permutation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs, out = ROOT / "runs", ROOT / "results"
    n, total = 0, 0
    for p in sorted(runs.rglob("*")):
        if not p.is_file() or p.suffix not in KEEP_SUFFIX:
            continue
        if p.stat().st_size > SIZE_LIMIT or any(s in p.name for s in SKIP_NAME_PARTS):
            continue
        dest = out / p.relative_to(runs)
        n += 1
        total += p.stat().st_size
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    print(f"{'would copy' if args.dry_run else 'copied'} {n} files, {total / 1e6:.2f} MB -> {out}")


if __name__ == "__main__":
    main()
