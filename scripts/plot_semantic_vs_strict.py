#!/usr/bin/env python3
"""
Minimal plot: strict ASR vs semantic leakage rate by (defense, variant).

Requires matplotlib (`pip install matplotlib`). If missing, prints CSV to stdout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _variant_from_payload(payload: Dict[str, Any]) -> str:
    meta = payload.get("attack_prompt_meta", {}) or {}
    variant = meta.get("variant_label") or meta.get("language_pair") or meta.get(
        "target_language"
    )
    if not variant:
        if meta.get("source_language") == "en" or not meta:
            return "EN"
        return "UNKNOWN"
    return str(variant)


def _gpt_level_rates(payload: Dict[str, Any]) -> Tuple[float, float, float]:
    """Returns (strict_success_rate, semantic_candidate_rate, mean_true_score)."""
    runs = payload.get("runs", [])
    if not runs:
        return 0.0, 0.0, 0.0
    strict_hits = 0
    sem_hits = 0
    scores: List[float] = []
    for r in runs:
        atk_info = r.get("atk_info", [])
        st = 0
        se = 0
        mx = 0.0
        for att in atk_info:
            if att.get("success") == 1:
                st = 1
            sc = att.get("semantic_chunk_leakage_v2") or {}
            if sc.get("error") is None:
                if sc.get("semantic_candidate") == 1:
                    se = 1
                v = sc.get("true_prompt_semantic_score")
                if isinstance(v, (int, float)):
                    mx = max(mx, float(v))
        strict_hits += st
        sem_hits += se
        scores.append(mx)
    n = len(runs)
    return strict_hits / n, sem_hits / n, sum(scores) / max(1, len(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="PNG path (default: results_dir/semantic_vs_strict.png)",
    )
    args = ap.parse_args()
    rd = Path(args.results_dir)
    rows: List[Tuple[str, str, str, float, float, float]] = []

    import json

    for fp in sorted(rd.glob("atk_*_def_*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        def_name = fp.name.split("_def_", 1)[1].rsplit(".json", 1)[0]
        variant = _variant_from_payload(payload)
        atk_id = fp.name.split("_def_", 1)[0]
        s_strict, s_sem, s_mx = _gpt_level_rates(payload)
        rows.append((atk_id, def_name, variant, s_strict, s_sem, s_mx))

    by_cond: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = defaultdict(
        list
    )
    for _, def_name, variant, s_strict, s_sem, s_mx in rows:
        by_cond[(def_name, variant)].append((s_strict, s_sem, s_mx))

    agg: List[Tuple[str, str, float, float, float]] = []
    for (def_name, variant), vals in sorted(by_cond.items()):
        n = len(vals)
        agg.append(
            (
                def_name,
                variant,
                sum(v[0] for v in vals) / n,
                sum(v[1] for v in vals) / n,
                sum(v[2] for v in vals) / n,
            )
        )

    w = csv.writer(sys.stdout)
    sem_hdr = "mean_semantic_candidate_rate"
    score_hdr = "mean_true_prompt_score"
    w.writerow(
        ["defense", "variant", "mean_strict_asr", sem_hdr, score_hdr]
    )
    for a in agg:
        w.writerow([a[0], a[1], f"{a[2]:.4f}", f"{a[3]:.4f}", f"{a[4]:.4f}"])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n(matplotlib not installed; CSV above is the full summary.)",
            file=sys.stderr,
        )
        return

    if not agg:
        return

    labels = [f"{d[:12]}\n{v}" for d, v, _, _, _ in agg]
    x = range(len(agg))
    strict_y = [t[2] for t in agg]
    sem_y = [t[3] for t in agg]
    fig, ax = plt.subplots(figsize=(max(8, len(agg) * 0.5), 4))
    w_bar = 0.35
    ax.bar([i - w_bar / 2 for i in x], strict_y, w_bar, label="Strict ROUGE ASR")
    sem_label = "Semantic candidate rate"
    ax.bar([i + w_bar / 2 for i in x], sem_y, w_bar, label=sem_label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("Strict vs semantic chunk leakage (per attack file → mean over GPTs)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    out = Path(args.out) if args.out else rd / "semantic_vs_strict.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
