"""Compliance checker — reads decisions.jsonl from the live agent.

Detects:
  - Auto-close overrides triggered by thesis tracker (rationale contains [AUTO-CLOSE)
  - thesis_strength distribution per asset (how often Claude scores 1/2/3/4/5)
  - Rule violations: buy/sell action when thesis_strength <= 3 (opening against entry rules)

Usage:
  python -m src.backtest.compliance_check
  python -m src.backtest.compliance_check --path /path/to/decisions.jsonl
"""

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))


def load_decisions(path: str) -> list[dict]:
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File not found: {path}")
    return entries


def analyse(entries: list[dict]):
    auto_close_count: dict[str, int] = defaultdict(int)
    score_dist: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    open_violations: dict[str, int] = defaultdict(int)
    total_cycles = len(entries)
    total_decisions = 0

    for entry in entries:
        for d in entry.get("decisions", []):
            asset = d.get("asset") or "unknown"
            action = d.get("action", "hold")
            rationale = d.get("rationale", "")
            ts = d.get("thesis_strength")
            total_decisions += 1

            if ts is not None:
                try:
                    score_dist[asset][int(ts)] += 1
                except (ValueError, TypeError):
                    pass

            if "[AUTO-CLOSE" in rationale:
                auto_close_count[asset] += 1

            # Opening against entry rules: buy/sell with thesis_strength <= 3
            # (thesis_strength 3 = neutral, rules say no new opens at 3 or below)
            if action in ("buy", "sell") and ts is not None:
                try:
                    if int(ts) <= 3:
                        open_violations[asset] += 1
                except (ValueError, TypeError):
                    pass

    return {
        "total_cycles": total_cycles,
        "total_decisions": total_decisions,
        "auto_close_count": dict(auto_close_count),
        "score_dist": {k: dict(v) for k, v in score_dist.items()},
        "open_violations": dict(open_violations),
    }


def print_report(results: dict):
    print(f"\n{'='*60}")
    print(f"Compliance Report")
    print(f"{'='*60}")
    print(f"Cycles analysed:   {results['total_cycles']}")
    print(f"Decisions logged:  {results['total_decisions']}")

    total_auto = sum(results["auto_close_count"].values())
    print(f"Auto-close total:  {total_auto}")

    assets = sorted(
        set(results["score_dist"]) | set(results["auto_close_count"]) | set(results["open_violations"])
    )
    if not assets:
        print("\nNo per-asset data found.")
        return

    for asset in assets:
        dist = results["score_dist"].get(asset, {})
        total_scored = sum(dist.values())
        auto = results["auto_close_count"].get(asset, 0)
        violations = results["open_violations"].get(asset, 0)

        print(f"\n  {asset}")
        if total_scored:
            avg = sum(k * v for k, v in dist.items()) / total_scored
            bar = "  ".join(f"{s}:{'█' * dist.get(s, 0)}{dist.get(s, 0)}" for s in range(1, 6))
            print(f"    Avg thesis_strength: {avg:.2f}   [{bar}]")
        else:
            print(f"    thesis_strength: no data (field not logged)")
        if auto:
            print(f"    Auto-close overrides: {auto}")
        if violations:
            print(f"    ⚠  Open violations (opened with thesis ≤ 3): {violations}")

    print(f"\n{'='*60}")
    total_violations = sum(results["open_violations"].values())
    if total_violations == 0 and total_auto == 0:
        print("No rule violations detected.")
    else:
        if total_violations:
            print(f"⚠  {total_violations} open-on-weak-thesis violation(s) detected.")
        if total_auto:
            print(f"ℹ  {total_auto} auto-close override(s) triggered by thesis tracker.")


def main():
    parser = argparse.ArgumentParser(description="Compliance check on decisions.jsonl")
    parser.add_argument("--path", default="decisions.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"decisions.jsonl not found at: {args.path}")
        print("Run the live agent first to generate decisions.jsonl.")
        sys.exit(1)

    entries = load_decisions(args.path)
    if not entries:
        print("No entries found.")
        sys.exit(0)

    results = analyse(entries)
    print_report(results)


if __name__ == "__main__":
    main()
