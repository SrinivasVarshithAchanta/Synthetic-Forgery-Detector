"""Build the final comparison report: baseline (CE) vs focal + rebalancing.

    python -m src.report --baseline reports/eval_baseline.json \
        --focal reports/eval_focal.json --out reports/results.json

Prints the measured before/after numbers (no assumptions) and writes:
  - reports/focal_delta.md      comparison table for the write-up
  - reports/results.json        final consolidated metrics (best of the two
                                models by subtle-FAR at equal FRR is chosen
                                as the `final_model`)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:.2f}%"


def delta(a: float, b: float) -> str:
    if a != a or b != b:
        return "n/a"
    return f"{(b - a) * 100:+.2f} pts"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--focal", default=None)
    ap.add_argument("--out", default="reports/results.json")
    ap.add_argument("--delta-md", default="reports/focal_delta.md")
    args = ap.parse_args()

    base = load(args.baseline)
    focal = load(args.focal) if args.focal and os.path.exists(args.focal) else None

    def row(label: str, b: float, f: float | None) -> str:
        return (f"| {label} | {pct(b)} | {pct(f) if f is not None else '-'} "
                f"| {delta(b, f) if f is not None else '-'} |")

    lines = ["# Focal loss / rebalancing: measured before & after", ""]
    lines.append(f"Baseline: `{base['checkpoint']}` "
                 f"(threshold {base['threshold']:.3f})")
    if focal:
        lines.append(f"Focal: `{focal['checkpoint']}` "
                     f"(threshold {focal['threshold']:.3f})")
    lines.append("")
    lines.append("| metric | baseline (CE) | focal + rebalance | delta |")
    lines.append("|---|---|---|---|")

    bs, fs = base["slices"]["clean"], (focal or base)["slices"]["clean"]
    rows = [
        ("overall test accuracy", bs["accuracy"], fs["accuracy"]),
        ("precision (tampered)", bs["precision"], fs["precision"]),
        ("recall (tampered)", bs["recall"], fs["recall"]),
        ("F1", bs["f1"], fs["f1"]),
        ("FAR (all tampered)", bs["far"], fs["far"]),
        ("FAR (subtle tampered)", base.get("far_subtle"),
         (focal or base).get("far_subtle")),
        ("FAR (aggressive tampered)", base.get("far_aggressive"),
         (focal or base).get("far_aggressive")),
        ("FRR (genuine rejected)", bs["frr"], fs["frr"]),
        ("AUROC", bs.get("auroc"), (focal or base).get("auroc")),
    ]
    for label, b, f in rows:
        lines.append(row(label, b, f))

    # bucket-level deltas (type x difficulty)
    if focal and base.get("far_by_bucket"):
        fb = {f"{b['forgery_type']}/{b['difficulty']}": b["far"]
              for b in base["far_by_bucket"]}
        fl = {f"{b['forgery_type']}/{b['difficulty']}": b["far"]
              for b in focal.get("far_by_bucket", [])}
        lines += ["", "| forgery bucket | baseline FAR | focal FAR | delta |",
                  "|---|---|---|---|"]
        for k in sorted(set(fb) & set(fl)):
            lines.append(f"| {k} | {pct(fb[k])} | {pct(fl[k])} "
                         f"| {delta(fb[k], fl[k])} |")

    # noisy / adversarial slices
    lines += ["", "| slice | accuracy (base) | accuracy (focal) | "
              "FAR (base) | FAR (focal) |", "|---|---|---|---|---|"]
    for sl in ["clean", "noisy", "adversarial"]:
        b = base["slices"].get(sl, {})
        f = (focal or base)["slices"].get(sl, {})
        lines.append(f"| {sl} | {pct(b.get('accuracy', float('nan')))} "
                     f"| {pct(f.get('accuracy', float('nan')))} "
                     f"| {pct(b.get('far', float('nan')))} "
                     f"| {pct(f.get('far', float('nan')))} |")

    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.delta_md) or ".", exist_ok=True)
    with open(args.delta_md, "w") as fh:
        fh.write(md)
    print(md)
    print(f"wrote {args.delta_md}")

    # consolidated results: keep both, mark the final model
    results = {"baseline": base,
               "focal": focal,
               "checkpoint": base["checkpoint"]}
    # choose final = lower subtle FAR (tie -> higher F1)
    if focal:
        if (focal.get("far_subtle", 1) < base.get("far_subtle", 1)
                or (focal["slices"]["clean"]["f1"]
                    > base["slices"]["clean"]["f1"])):
            results["final_model"] = "focal"
            results["checkpoint"] = focal["checkpoint"]
        else:
            results["final_model"] = "baseline"
    else:
        results["final_model"] = "baseline"
    final = base if results["final_model"] == "baseline" else focal
    results["slices"] = final["slices"]
    results["far_subtle"] = final.get("far_subtle")
    results["far_aggressive"] = final.get("far_aggressive")
    results["far_by_bucket"] = final.get("far_by_bucket")
    results["threshold"] = final["threshold"]
    results["overall_test_accuracy"] = final.get("overall_test_accuracy",
                                                 final["slices"]["clean"]["accuracy"])
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"final model: {results['final_model']} -> wrote {args.out}")


if __name__ == "__main__":
    main()
