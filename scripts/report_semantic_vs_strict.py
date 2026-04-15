#!/usr/bin/env python3
"""
Compare strict ROUGE-L ASR to semantic_metric_v2 (semantic_chunk_leakage_v2).

Usage:
  python scripts/report_semantic_vs_strict.py --results_dir results/<run_name>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _percentile_sorted(xs_sorted: List[float], p: float) -> float:
    if not xs_sorted:
        return 0.0
    i = int(round(p * (len(xs_sorted) - 1)))
    return xs_sorted[max(0, min(i, len(xs_sorted) - 1))]


def _collect_rows_v2(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in payload.get("runs", []):
        ai = r.get("atk_info", [])
        strict = 1 if any(a.get("success") == 1 for a in ai) else 0
        sem, true_score, margin, max_neg = 0, 0.0, 0.0, 0.0
        detail: Dict[str, Any] = {}
        has_semantic = False
        for a in ai:
            sc = a.get("semantic_chunk_leakage_v2")
            if not isinstance(sc, dict) or sc.get("error"):
                continue
            has_semantic = True
            if sc.get("semantic_candidate") == 1:
                sem = 1
            ts = sc.get("true_prompt_semantic_score")
            if isinstance(ts, (int, float)):
                true_score = max(true_score, float(ts))
            mg = sc.get("semantic_margin")
            if isinstance(mg, (int, float)):
                margin = max(margin, float(mg))
            mn = sc.get("max_negative_prompt_score")
            if isinstance(mn, (int, float)):
                max_neg = max(max_neg, float(mn))
            detail = sc
        rows.append(
            {
                "strict": strict,
                "sem": sem,
                "max_sim": true_score,
                "margin": margin,
                "max_neg": max_neg,
                "has_semantic": has_semantic,
                "detail": detail,
            }
        )
    return rows


def _file_has_v2(payload: Dict[str, Any]) -> bool:
    for r in payload.get("runs", [])[:5]:
        for a in r.get("atk_info", []):
            sc = a.get("semantic_chunk_leakage_v2")
            if isinstance(sc, dict) and not sc.get("error"):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", type=str, required=True)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    if not rd.is_dir():
        raise SystemExit(f"Not a directory: {rd}")

    files = sorted(rd.glob("atk_*_def_*.json"))
    if not files:
        raise SystemExit(f"No atk_*_def_*.json under {rd}")

    with open(files[0], encoding="utf-8") as f:
        sample = json.load(f)
    if not _file_has_v2(sample):
        print(f"=== {rd} ===")
        print("No usable semantic_chunk_leakage_v2 in sample file.")
        print(
            "Run: python scripts/backfill_semantic_metrics.py "
            "--results_dir ... --in_place"
        )
        return

    all_strict: List[int] = []
    all_sem: List[int] = []
    all_max: List[float] = []
    all_margin: List[float] = []
    all_max_neg: List[float] = []
    all_mean_chunk: List[float] = []
    all_topk: List[float] = []
    all_frac: List[float] = []
    all_n_prompt: List[int] = []
    all_n_resp: List[int] = []
    n_no_semantic = 0
    per_file: List[Tuple[str, float, float, float, float]] = []

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            payload = json.load(f)
        rows = _collect_rows_v2(payload)
        if not rows:
            continue
        if not any(r["has_semantic"] for r in rows):
            n_no_semantic += 1

        fs = [r["strict"] for r in rows]
        fsem = [r["sem"] for r in rows]
        fmx = [r["max_sim"] for r in rows]
        all_strict.extend(fs)
        all_sem.extend(fsem)
        all_max.extend(fmx)
        fmrg = [float(r.get("margin", 0.0)) for r in rows]
        fnv = [float(r.get("max_neg", 0.0)) for r in rows]
        all_margin.extend(fmrg)
        all_max_neg.extend(fnv)
        per_file.append(
            (
                fp.name,
                _mean([float(x) for x in fs]),
                _mean([float(x) for x in fsem]),
                _mean(fmx),
                _mean(fmrg),
            )
        )

        for r in rows:
            d = r["detail"]
            if not d or d.get("error"):
                continue
            if "mean_best_prompt_to_response_similarity" in d:
                all_mean_chunk.append(
                    float(d["mean_best_prompt_to_response_similarity"])
                )
            if "topk_pair_mean_similarity" in d:
                all_topk.append(float(d["topk_pair_mean_similarity"]))
            fa = d.get("diagnostic_fraction_prompt_chunks_above_threshold")
            if isinstance(fa, (int, float)):
                all_frac.append(float(fa))
            npi = d.get("num_prompt_chunks")
            nri = d.get("num_response_chunks")
            if isinstance(npi, int):
                all_n_prompt.append(npi)
            if isinstance(nri, int):
                all_n_resp.append(nri)

    n = len(all_strict)
    print(f"=== {rd} ===")
    print("Semantic report: pairwise chunks + negative controls (semantic_chunk_leakage_v2)")
    print(f"Attack JSON files: {len(files)}")
    print(f"GPT samples (rows): {n}")

    if n_no_semantic == len(files):
        print("\nNo usable semantic blocks. Run:")
        print(
            "  python scripts/backfill_semantic_metrics.py --results_dir ... --in_place"
        )
        print(f"\nStrict ROUGE-L ASR (per-GPT): {_mean([float(x) for x in all_strict]):.4f}")
        return

    print("\n--- Primary (strict ROUGE) vs secondary semantic ---")
    print(f"Mean strict ASR (per-GPT):        {_mean([float(x) for x in all_strict]):.4f}")
    print(f"Mean semantic_candidate rate:     {_mean([float(x) for x in all_sem]):.4f}")
    print(f"Mean true_prompt_semantic_score:   {_mean(all_max):.4f}")
    if all_margin:
        print(f"Mean semantic_margin:             {_mean(all_margin):.4f}")
        print(f"Mean max_negative_prompt_score:   {_mean(all_max_neg):.4f}")
    if all_mean_chunk:
        print(f"Mean auxiliary mean_* similarity: {_mean(all_mean_chunk):.4f}")
    if all_topk:
        print(f"Mean top-k pair similarity:       {_mean(all_topk):.4f}")
    if all_frac:
        print(f"Mean diagnostic frac prompt ch.: {_mean(all_frac):.4f}")
    if all_n_prompt:
        print(
            f"Mean num_prompt_chunks:           {_mean([float(x) for x in all_n_prompt]):.1f} "
            f"(min {min(all_n_prompt)}, max {max(all_n_prompt)})"
        )
    if all_n_resp:
        print(
            f"Mean num_response_chunks:         {_mean([float(x) for x in all_n_resp]):.1f} "
            f"(min {min(all_n_resp)}, max {max(all_n_resp)})"
        )

    both = sum(1 for s, e in zip(all_strict, all_sem) if s and e)
    strict_only = sum(1 for s, e in zip(all_strict, all_sem) if s and not e)
    sem_only = sum(1 for s, e in zip(all_strict, all_sem) if not s and e)
    neither = sum(1 for s, e in zip(all_strict, all_sem) if not s and not e)

    print("\n--- Per-GPT contingency (strict vs semantic_candidate) ---")
    print(f"strict=1 sem=1:  {both:5d} ({100 * both / n:.1f}%)")
    print(f"strict=1 sem=0:  {strict_only:5d} ({100 * strict_only / n:.1f}%)")
    print(f"strict=0 sem=1:  {sem_only:5d} ({100 * sem_only / n:.1f}%)")
    print(f"strict=0 sem=0:  {neither:5d} ({100 * neither / n:.1f}%)")

    print("\n--- Per-attack file (mean over GPTs) ---")
    for tup in per_file:
        name, a, b, c, mrg = tup
        print(
            f"  {name}: strict={a:.3f} candidate={b:.3f} "
            f"mean_true_score={c:.3f} mean_margin={mrg:.3f}"
        )

    m1 = sorted([m for m, e in zip(all_max, all_sem) if e == 1])
    m0 = sorted([m for m, e in zip(all_max, all_sem) if e == 0])
    if m1:
        print(
            f"\ntrue_score when sem=1: n={len(m1)} mean={_mean(m1):.3f} "
            f"median={statistics.median(m1):.3f} p90={_percentile_sorted(m1, 0.9):.3f}"
        )
    if m0:
        print(
            f"true_score when sem=0: n={len(m0)} mean={_mean(m0):.3f} "
            f"median={statistics.median(m0):.3f} p90={_percentile_sorted(m0, 0.9):.3f}"
        )

    if all_margin:
        mar_sorted = sorted(all_margin)
        print(
            f"\nsemantic_margin: mean={_mean(all_margin):.3f} "
            f"median={statistics.median(all_margin):.3f} "
            f"p10={_percentile_sorted(mar_sorted, 0.1):.3f} "
            f"p90={_percentile_sorted(mar_sorted, 0.9):.3f}"
        )


if __name__ == "__main__":
    main()
