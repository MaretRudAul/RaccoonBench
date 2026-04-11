import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple


def _attack_success_from_runs(payload: Dict[str, Any]) -> float:
    """
    Compute success rate over GPTs for a single attack+defense json file.

    Each run has atk_info as a list of attempts; success is 1 if any attempt succeeded.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help="Path like results/run_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("atk_*_def_*.json"))
    if not files:
        raise SystemExit(f"No result files found in {results_dir}")

    by_def_and_variant: Dict[Tuple[str, str], list] = defaultdict(list)

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload.get("attack_prompt_meta", {}) or {}
        variant = meta.get("variant_label") or meta.get("language_pair") or meta.get("target_language")
        if not variant:
            # English-only runs may only store minimal metadata. Treat them as EN for readability.
            # This also keeps older/sparser result files analyzable without re-running.
            if meta.get("source_language") == "en" or not meta:
                variant = "EN"
            else:
                variant = "UNKNOWN"

        # derive defense name from filename suffix to match saved layout
        def_name = fp.name.split("_def_", 1)[1].rsplit(".json", 1)[0]
        rate = _attack_success_from_runs(payload)
        by_def_and_variant[(def_name, str(variant))].append(rate)

    print("Average success rate (mean over attacks) by defense and variant:")
    for (def_name, variant), rates in sorted(by_def_and_variant.items()):
        mean_rate = sum(rates) / max(1, len(rates))
        print(f"- defense={def_name:20s} variant={variant:6s} mean_success={mean_rate:.3f} (n_attacks={len(rates)})")


if __name__ == "__main__":
    main()

