import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _collect_atk_json_files(
    results_dir: Path,
) -> Tuple[List[Path], Dict[Path, str]]:
    """
    Discover result JSON files.

    - If ``results_dir`` contains ``atk_*_def_*.json`` directly (legacy / flat layout),
      return those with an empty mode label per file.
    - Otherwise, scan **immediate child subdirectories** only; for each subdir that
      contains matching JSON, attach that folder name as the run mode (e.g. three-mode
      benchmark: undefended/, translate_attack_to_english/, undefended_system_security_suffix/).
    """
    direct = sorted(results_dir.glob("atk_*_def_*.json"))
    if direct:
        return direct, {p: "" for p in direct}

    out_files: List[Path] = []
    path_to_mode: Dict[Path, str] = {}
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        found = sorted(sub.glob("atk_*_def_*.json"))
        if not found:
            continue
        for p in found:
            out_files.append(p)
            path_to_mode[p] = sub.name
    return out_files, path_to_mode


def _mode_from_payload_or_path(
    payload: Dict[str, Any], fp: Path, fallback_dir_mode: str
) -> str:
    """Prefer JSON benchmark_condition; else subdirectory name used during discovery."""
    bc = payload.get("benchmark_condition")
    if isinstance(bc, str) and bc.strip():
        return bc.strip()
    return fallback_dir_mode


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


def _files_have_semantic_v2(files: List[Path]) -> bool:
    """
    True if any result file contains semantic_chunk_leakage_v2 in atk_info.

    Scans all files: multi-mode runs may list incomplete subfolders first.
    """
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for r in payload.get("runs", []):
            for att in r.get("atk_info", []):
                if isinstance(att.get("semantic_chunk_leakage_v2"), dict):
                    return True
    return False


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
        help="Path like results/run_YYYYMMDD_HHMMSS (flat JSON) or a run directory whose "
        "JSON files live in immediate subfolders (e.g. undefended/, translate_attack_to_english/, "
        "undefended_system_security_suffix/).",
    )
    ap.add_argument(
        "--semantic",
        action="store_true",
        help="Also report semantic_chunk_leakage_v2 aggregates (if present in JSON).",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    files, path_to_mode = _collect_atk_json_files(results_dir)
    if not files:
        raise SystemExit(
            f"No result files found in {results_dir} (expected atk_*_def_*.json here "
            f"or in immediate subdirectories)."
        )

    use_subdir_modes = any(path_to_mode.get(p, "") for p in files)

    AggregateKey = Tuple[str, str, str]
    by_key: Dict[AggregateKey, list] = defaultdict(list)
    by_key_sem: Dict[AggregateKey, list] = defaultdict(list)
    by_key_mx: Dict[AggregateKey, list] = defaultdict(list)
    by_key_nsem: Dict[AggregateKey, list] = defaultdict(list)
    by_key_mrg: Dict[AggregateKey, list] = defaultdict(list)
    by_key_any_runs: Dict[AggregateKey, bool] = defaultdict(bool)
    have_semantic_v2 = bool(args.semantic and files and _files_have_semantic_v2(files))

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        dir_mode = path_to_mode.get(fp, "")
        mode = (
            _mode_from_payload_or_path(payload, fp, dir_mode)
            if use_subdir_modes
            else ""
        )
        variant = _variant_from_payload(payload)
        def_name = fp.name.split("_def_", 1)[1].rsplit(".json", 1)[0]
        rate = _attack_success_from_runs(payload)
        key: AggregateKey = (mode, def_name, variant)
        by_key[key].append(rate)
        if payload.get("runs"):
            by_key_any_runs[key] = True

        if args.semantic and have_semantic_v2:
            sr, mx, mrg, nsem = _semantic_v2_from_runs(payload)
            by_key_sem[key].append(sr)
            by_key_mx[key].append(mx)
            by_key_mrg[key].append(mrg)
            by_key_nsem[key].append(nsem)

    print("Average success rate (mean over attacks) by defense and variant:")
    print("(Primary metric: strict ROUGE-L ASR, threshold unchanged in pipeline)")
    if use_subdir_modes:
        print(
            f"(Multiple run modes from subdirectories of {results_dir}; "
            "mode= from JSON benchmark_condition when present, else folder name.)"
        )
    for (mode, def_name, variant), rates in sorted(by_key.items()):
        mean_rate = sum(rates) / max(1, len(rates))
        if use_subdir_modes:
            line = (
                f"- mode={mode:32s} defense={def_name:20s} variant={variant:6s} "
                f"mean_strict_asr={mean_rate:.3f} (n_attacks={len(rates)})"
            )
        else:
            line = f"- defense={def_name:20s} variant={variant:6s} mean_strict_asr={mean_rate:.3f} (n_attacks={len(rates)})"
        if args.semantic:
            sem_rates = by_key_sem.get((mode, def_name, variant), [])
            mx_rates = by_key_mx.get((mode, def_name, variant), [])
            nsem = by_key_nsem.get((mode, def_name, variant), [0])
            if have_semantic_v2 and sem_rates and max(nsem) > 0:
                ms = sum(sem_rates) / len(sem_rates)
                mm = sum(mx_rates) / len(mx_rates)
                mrs = by_key_mrg.get((mode, def_name, variant), [])
                mrg = sum(mrs) / len(mrs) if mrs else 0.0
                line += (
                    f" | semantic_candidate_rate={ms:.3f} "
                    f"mean_true_score={mm:.3f} mean_margin={mrg:.3f}"
                )
            elif args.semantic:
                if not by_key_any_runs.get((mode, def_name, variant), False):
                    line += (
                        " | (no semantic summary: saved JSON has empty "
                        "\"runs\" — benchmark did not record GPTs for this row; "
                        "re-run that mode or check logs, not a backfill issue)"
                    )
                elif not have_semantic_v2:
                    line += " | (no semantic_chunk_leakage_v2 in any result file under this run)"
                elif sem_rates and max(nsem) == 0:
                    line += (
                        " | (runs present but no usable semantic_chunk_leakage_v2 in atk_info; "
                        "python scripts/backfill_semantic_metrics.py --results_dir ... --in_place)"
                    )
                else:
                    line += " | (semantic row skipped: no aggregated rates for this key)"
        print(line)


if __name__ == "__main__":
    main()
