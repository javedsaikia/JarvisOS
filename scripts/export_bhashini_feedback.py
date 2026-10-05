#!/usr/bin/env python3
"""Copy Assamese Bhashini feedback log for sharing with Bhashini / Bhasha Daan."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bella import bhashini_feedback  # noqa: E402


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out = bhashini_feedback.export_records(dest)
    count = bhashini_feedback.export_summary()
    print(f"Exported {count} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
