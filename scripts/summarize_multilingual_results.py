import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _variant_from_payload(payload: Dict[str, Any]) -> str:
    meta = payload.get("attack_prompt_meta", {}) or {}
    variant = meta.get("variant_label") or meta.get("language_pair") or meta.get(
        "target_language"
    )
    if not variant:
        if meta.get("source_language") == "en" or not meta:
            variant = "EN"
        else:
            variant = "UNKNOWN"
    return str(variant)


def _attack_success_from_runs(payload: Dict[str, Any]) -> float:
    runs = payload.get("runs", [])
    if not runs:
        return 0.0
    successes = 0
    for r in runs:
        atk_info = r.get("atk_info", [])
        any_success = 0
        for attempt in atk_info:
            if attempt.get("success") == 1:
                any_success = 1
                break
        successes += any_success
    return successes / max(1, len(runs))


def _semantic_from_runs(payload: Dict[str, Any]) -> Tuple[float, float, int]:
    """
    Per-GPT: max semantic_leakage_success across attempts, max max_chunk_similarity.
    Returns (mean_semantic_rate, mean_max_chunk_sim, n_gpts_with_semantic_field).
    """
    runs = payload.get("runs", [])
    if not runs:
        return 0.0, 0.0, 0
    sem_hits = 0
    max_sims: List[float] = []
    n_with = 0
    for r in runs:
        atk_info = r.get("atk_info", [])
        hit = 0
        mx = 0.0
        saw = False
        for att in atk_info:
            sc = att.get("semantic_chunk_leakage")
            if not isinstance(sc, dict):
                continue
            saw = True
            if sc.get("error") is None:
                if sc.get("semantic_leakage_success") == 1:
                    hit = 1
                v = sc.get("max_chunk_similarity")
                if isinstance(v, (int, float)):
                    mx = max(mx, float(v))
        if saw:
            n_with += 1
        sem_hits += hit
        max_sims.append(mx)
    return (
        sem_hits / max(1, len(runs)),
        sum(max_sims) / max(1, len(max_sims)),
        n_with,
    )


def _semantic_v2_from_runs(
    payload: Dict[str, Any],
) -> Tuple[float, float, float, int]:
    """Returns (candidate_rate, mean_true_score, mean_margin, n_with_field)."""
    runs = payload.get("runs", [])
    if not runs:
        return 0.0, 0.0, 0.0, 0
    hits = 0
    true_scores: List[float] = []
    margins: List[float] = []
    n_with = 0
    for r in runs:
        atk_info = r.get("atk_info", [])
        hit = 0
        ts, mg = 0.0, 0.0
        saw = False
        for att in atk_info:
            sc = att.get("semantic_chunk_leakage_v2")
            if not isinstance(sc, dict):
                continue
            saw = True
            if sc.get("error") is None:
                if sc.get("semantic_candidate") == 1:
                    hit = 1
                v = sc.get("true_prompt_semantic_score")
                if isinstance(v, (int, float)):
                    ts = max(ts, float(v))
                m = sc.get("semantic_margin")
                if isinstance(m, (int, float)):
                    mg = max(mg, float(m))
        if saw:
            n_with += 1
        hits += hit
        true_scores.append(ts)
        margins.append(mg)
    return (
        hits / max(1, len(runs)),
        sum(true_scores) / max(1, len(true_scores)),
        sum(margins) / max(1, len(margins)),
        n_with,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path like results/run_YYYYMMDD_HHMMSS",
    )
    ap.add_argument(
        "--semantic",
        action="store_true",
        help="Also report semantic_chunk_leakage aggregates (if present in JSON).",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("atk_*_def_*.json"))
    if not files:
        raise SystemExit(f"No result files found in {results_dir}")

    by_def_and_variant: Dict[Tuple[str, str], list] = defaultdict(list)
    by_def_and_variant_sem: Dict[Tuple[str, str], list] = defaultdict(list)
    by_def_and_variant_mx: Dict[Tuple[str, str], list] = defaultdict(list)
    by_def_and_variant_nsem: Dict[Tuple[str, str], list] = defaultdict(list)
    by_def_and_variant_mrg: Dict[Tuple[str, str], list] = defaultdict(list)
    use_v2 = False
    if args.semantic and files:
        with open(files[0], encoding="utf-8") as f:
            sample = json.load(f)
        for r in sample.get("runs", [])[:3]:
            for att in r.get("atk_info", []):
                if isinstance(att.get("semantic_chunk_leakage_v2"), dict):
                    use_v2 = True
                    break
            if use_v2:
                break

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        variant = _variant_from_payload(payload)
        def_name = fp.name.split("_def_", 1)[1].rsplit(".json", 1)[0]
        rate = _attack_success_from_runs(payload)
        by_def_and_variant[(def_name, variant)].append(rate)

        if args.semantic:
            if use_v2:
                sr, mx, mrg, nsem = _semantic_v2_from_runs(payload)
                by_def_and_variant_sem[(def_name, variant)].append(sr)
                by_def_and_variant_mx[(def_name, variant)].append(mx)
                by_def_and_variant_mrg[(def_name, variant)].append(mrg)
                by_def_and_variant_nsem[(def_name, variant)].append(nsem)
            else:
                sr, mx, nsem = _semantic_from_runs(payload)
                by_def_and_variant_sem[(def_name, variant)].append(sr)
                by_def_and_variant_mx[(def_name, variant)].append(mx)
                by_def_and_variant_nsem[(def_name, variant)].append(nsem)

    print("Average success rate (mean over attacks) by defense and variant:")
    print("(Primary metric: strict ROUGE-L ASR, threshold unchanged in pipeline)")
    for (def_name, variant), rates in sorted(by_def_and_variant.items()):
        mean_rate = sum(rates) / max(1, len(rates))
        line = f"- defense={def_name:20s} variant={variant:6s} mean_strict_asr={mean_rate:.3f} (n_attacks={len(rates)})"
        if args.semantic:
            sem_rates = by_def_and_variant_sem.get((def_name, variant), [])
            mx_rates = by_def_and_variant_mx.get((def_name, variant), [])
            nsem = by_def_and_variant_nsem.get((def_name, variant), [0])
            if sem_rates and max(nsem) > 0:
                ms = sum(sem_rates) / len(sem_rates)
                mm = sum(mx_rates) / len(mx_rates)
                if use_v2:
                    mrs = by_def_and_variant_mrg.get((def_name, variant), [])
                    mrg = sum(mrs) / len(mrs) if mrs else 0.0
                    line += f" | v2 candidate_rate={ms:.3f} mean_true_score={mm:.3f} mean_margin={mrg:.3f}"
                else:
                    line += f" | mean_semantic_leak_rate={ms:.3f} mean_max_chunk_sim={mm:.3f}"
            else:
                line += " | (no semantic fields; backfill_semantic_metrics_v2.py or v1 backfill)"
        print(line)


if __name__ == "__main__":
    main()
