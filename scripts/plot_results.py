#!/usr/bin/env python3
"""
Generate multilingual benchmark figures from a combined Excel summary.

Default data file: ``multilingual_results.xlsx`` in the repository root (sheet
``multilingual_summary``), produced by merging per-run ``multilingual_summary.xlsx``
exports or equivalent.

Figures:
  1. Mean ASR by language variant, faceted by model, grouped by defense condition.
  2. Semantic candidate rate, same layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Panel order (must match rows in ``multilingual_results.xlsx`` for standard runs)
MODEL_ORDER: List[Tuple[str, str]] = [
    ("gpt-3.5-turbo-0125", "GPT-3.5-turbo-0125"),
    ("gpt-5.4-nano", "GPT-5.4-nano"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B Instruct"),
]

VARIANT_ORDER = ["EN", "BN", "ZU", "BN+ZU"]

# Canonical defense keys and plot order (three bars per language variant)
DEFENSE_SPEC: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("undefended", "Undefended", ("undefended",)),
    (
        "translate_english",
        "Translate-to-English",
        ("translate_to_english_defense", "translate_attack_to_english"),
    ),
    (
        "system_security_suffix",
        "System-security-suffix",
        ("undefended_system_security_suffix",),
    ),
]


def _as_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return float("nan")
    return float(s)


def _normalize_mode(mode: Any) -> str | None:
    """Map raw ``mode`` cell to a canonical defense key."""
    if mode is None:
        return None
    m = str(mode).strip()
    for canon, _label, raw_names in DEFENSE_SPEC:
        if m in raw_names:
            return canon
    return None


def _display_title_for_model(model_id: str) -> str:
    for mid, title in MODEL_ORDER:
        if mid == model_id:
            return title
    return model_id


def load_rows(xlsx_path: Path, sheet_name: str | None) -> List[Dict[str, Any]]:
    try:
        import openpyxl
    except ImportError as e:
        raise SystemExit(
            "plot_results.py requires openpyxl. Install with: pip install openpyxl"
        ) from e

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name:
        ws = wb[sheet_name]
    else:
        ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        wb.close()
        return []
    names = [str(h).strip() if h is not None else "" for h in header]
    out: List[Dict[str, Any]] = []
    for tup in rows_iter:
        if not tup or all(x is None for x in tup):
            continue
        row = {names[i]: tup[i] for i in range(len(names)) if i < len(tup)}
        out.append(row)
    wb.close()
    return out


def build_lookup(
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Dict[str, Tuple[float, float]]]], List[str]]:
    """
    Returns (lookup, model_ids_in_order).

    lookup[model_id][defense_canon][variant] = (mean_ASR, semantic_candidate_rate)
    """
    # model -> defense -> variant -> (asr, sem)
    acc: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    seen_models: List[str] = []

    for row in rows:
        mid = row.get("model")
        if mid is None or str(mid).strip() == "":
            continue
        model_id = str(mid).strip()
        if model_id not in seen_models:
            seen_models.append(model_id)

        canon = _normalize_mode(row.get("mode"))
        if canon is None:
            continue

        var = row.get("variant")
        if var is None:
            continue
        variant = str(var).strip()
        if variant not in VARIANT_ORDER:
            continue

        asr = _as_float(row.get("mean_ASR"))
        sem = _as_float(row.get("semantic_candidate_rate"))

        acc.setdefault(model_id, {}).setdefault(canon, {})[variant] = (asr, sem)

    # Order panels: MODEL_ORDER first, then any extra models from file
    ordered: List[str] = []
    for mid, _ in MODEL_ORDER:
        if mid in acc and mid not in ordered:
            ordered.append(mid)
    for mid in seen_models:
        if mid in acc and mid not in ordered:
            ordered.append(mid)

    return acc, ordered


def _plot_faceted_grouped_bars(
    lookup: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    model_ids: List[str],
    *,
    value_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise SystemExit(
            "plot_results.py requires matplotlib. Install with: pip install matplotlib"
        ) from e

    n_panels = len(model_ids)
    if n_panels == 0:
        raise SystemExit("No model rows to plot.")

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.2 * n_panels, 4.0),
        sharey=True,
        squeeze=False,
    )
    ax_flat = axes[0]

    n_variants = len(VARIANT_ORDER)
    n_def = len(DEFENSE_SPEC)
    x = np.arange(n_variants, dtype=float)
    width = 0.22
    offsets = np.linspace(-(n_def - 1) / 2 * width, (n_def - 1) / 2 * width, n_def)

    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for col, model_id in enumerate(model_ids):
        ax = ax_flat[col]
        mdata = lookup.get(model_id, {})
        for di, ((_canon, label, _raw), color) in enumerate(
            zip(DEFENSE_SPEC, colors)
        ):
            ys: List[float] = []
            for v in VARIANT_ORDER:
                cell = mdata.get(_canon, {}).get(v)
                if cell is None:
                    ys.append(float("nan"))
                else:
                    asr, sem = cell
                    ys.append(asr if value_key == "asr" else sem)
            ax.bar(
                x + offsets[di],
                ys,
                width,
                label=label if col == 0 else None,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_ORDER)
        ax.set_title(_display_title_for_model(model_id), fontsize=11)
        ax.set_xlabel("Language variant")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    ax_flat[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=12, y=1.02)
    handles, labels = ax_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        plt.subplots_adjust(bottom=0.22)
    else:
        plt.subplots_adjust(bottom=0.12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot mean ASR and semantic candidate rate from multilingual_results.xlsx."
    )
    ap.add_argument(
        "--xlsx",
        type=str,
        default=str(ROOT / "multilingual_results.xlsx"),
        help="Path to combined Excel file (default: repo root multilingual_results.xlsx).",
    )
    ap.add_argument(
        "--sheet",
        type=str,
        default="",
        help="Worksheet name (default: active sheet).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "figures"),
        help="Directory for PNG outputs (default: figures/).",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI (default: 200).")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    if not xlsx_path.is_file():
        raise SystemExit(f"Excel file not found: {xlsx_path}")

    sheet = args.sheet.strip() or None
    rows = load_rows(xlsx_path, sheet)
    if not rows:
        raise SystemExit(f"No data rows in {xlsx_path}")

    lookup, model_ids = build_lookup(rows)
    if not model_ids:
        raise SystemExit("No usable (model, mode, variant) rows after parsing.")

    out_dir = Path(args.out_dir).expanduser().resolve()

    _plot_faceted_grouped_bars(
        lookup,
        model_ids,
        value_key="asr",
        ylabel="Mean ASR",
        title="Mean ASR by language variant, defense condition, and model",
        out_path=out_dir / "mean_asr_by_variant.png",
        dpi=args.dpi,
    )
    _plot_faceted_grouped_bars(
        lookup,
        model_ids,
        value_key="sem",
        ylabel="Semantic candidate rate",
        title="Semantic candidate rate by language variant, defense condition, and model",
        out_path=out_dir / "semantic_candidate_rate_by_variant.png",
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
