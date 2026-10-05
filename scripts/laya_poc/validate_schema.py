#!/usr/bin/env python3
"""Validate hierarchical routing schema token budgets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bella.laya_router import COARSE_CATEGORIES, FINE_TOOLS, head_max_len_report  # noqa: E402


def main() -> int:
    report = head_max_len_report()
    print("Hierarchical schema validation\n")
    print(f"Coarse categories: {len(COARSE_CATEGORIES)}")
    print(f"  budget @ head_max_len=256: {report['coarse']}")
    print(f"\nFine stage max options: {report['fine_max']}")
    for cat, info in report["fine_worst"].items():
        print(f"  worst category {cat!r}: {info}")

    ok = (
        report["coarse"]["ok"]
        and report["coarse"]["options"] <= 8
        and report["fine_max"] <= 15
    )
    out = Path(__file__).parent / "schema_validation.json"
    out.write_text(json.dumps({"ok": ok, **report}, indent=2) + "\n")
    print(f"\nSchema within Laya budgets: {ok}")
    print(f"Wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
