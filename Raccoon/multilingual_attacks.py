import logging
from typing import List, Dict, Any, Optional, Set

from Raccoon.prompt import AttPrompt
from Raccoon.attack_output_language import append_english_output_instruction
from Raccoon.translation_utils import AttackTranslator, split_attack_text


SUPPORTED_LANGS = ["en", "bn", "zu"]
MIXED_PAIR = ("bn", "zu")

ALL_MULTILINGUAL_VARIANTS: Set[str] = {"EN", "BN", "ZU", "BN+ZU"}
_VARIANTS_NEEDING_TRANSLATION: Set[str] = {"BN", "ZU", "BN+ZU"}


def parse_multilingual_variant_filter(spec: str) -> Set[str]:
    """
    Parse comma-separated variant names (case-insensitive).

    Allowed tokens: en, bn, zu, bn+zu (also bnzu or mixed for BN+ZU).
    """
    if not spec or not spec.strip():
        return set(ALL_MULTILINGUAL_VARIANTS)
    out: Set[str] = set()
    for raw in spec.split(","):
        p = raw.strip().lower().replace(" ", "")
        if not p:
            continue
        if p == "en":
            out.add("EN")
        elif p == "bn":
            out.add("BN")
        elif p == "zu":
            out.add("ZU")
        elif p in ("bn+zu", "bnzu", "mixed"):
            out.add("BN+ZU")
        else:
            raise ValueError(
                f"Unknown multilingual variant {raw!r}. "
                f"Use comma-separated: en, bn, zu, bn+zu (got: {spec!r})"
            )
    if not out:
        raise ValueError(f"Empty multilingual variant list: {spec!r}")
    unknown = out - ALL_MULTILINGUAL_VARIANTS
    if unknown:
        raise ValueError(f"Invalid variant set: {unknown}")
    return out


def _variant_label(meta: Dict[str, Any]) -> str:
    if meta.get("variant_type") == "mixed":
        return "BN+ZU"
    lang = meta.get("target_language")
    if lang == "en":
        return "EN"
    if lang == "bn":
        return "BN"
    if lang == "zu":
        return "ZU"
    return str(lang).upper()


def expand_attack_prompts_multilingual(
    base_attacks: List[AttPrompt],
    translator: Optional[AttackTranslator],
    variant_filter: Optional[Set[str]] = None,
) -> List[AttPrompt]:
    """
    Expand each base English attack into a subset of:
    - mono EN (original)
    - mono BN (LLM translation)
    - mono ZU (LLM translation)
    - mixed BN+ZU (deterministic split + translation)

    variant_filter: subset of {"EN","BN","ZU","BN+ZU"}; default = all four.
    If translator is None, only "EN" may appear in variant_filter.
    """
    wanted = set(variant_filter) if variant_filter is not None else set(ALL_MULTILINGUAL_VARIANTS)
    need_tr = wanted & _VARIANTS_NEEDING_TRANSLATION
    if need_tr and translator is None:
        raise ValueError(
            f"Translator required for variants {sorted(need_tr)} but translator was None."
        )

    expanded: List[AttPrompt] = []

    for atk in base_attacks:
        base_text = atk.get_att_prompt()
        base_name = getattr(atk, "name", None) or atk.get_metadata().get("base_attack_name") or "attack"

        # EN mono (canonical)
        if "EN" in wanted:
            meta_en: Dict[str, Any] = {
                "base_attack_name": base_name,
                "variant_type": "mono",
                "source_language": "en",
                "target_language": "en",
                "target_languages": None,
                "language_pair": None,
                "split_method": None,
                "translation_model": None,
                "translation_prompt_version": None,
                "variant_label": "EN",
            }
            expanded.append(
                AttPrompt(
                    att_prompt=base_text,
                    category=atk.category,
                    name=base_name,
                    metadata=meta_en,
                )
            )

        if translator is None:
            continue

        # BN mono
        if "BN" in wanted:
            bn_res = translator.translate_attack(base_text, "bn")
            if bn_res is None:
                logging.error(f"Skipping BN variant for base attack {base_name} (translation failed).")
            else:
                meta_bn: Dict[str, Any] = {
                    "base_attack_name": base_name,
                    "variant_type": "mono",
                    "source_language": "en",
                    "target_language": "bn",
                    "target_languages": None,
                    "language_pair": None,
                    "split_method": None,
                    "translation_model": bn_res.translation_model,
                    "translation_prompt_version": bn_res.translation_prompt_version,
                    "variant_label": "BN",
                }
                expanded.append(
                    AttPrompt(
                        att_prompt=bn_res.text,
                        category=atk.category,
                        name=base_name,
                        metadata=meta_bn,
                    )
                )

        # ZU mono
        if "ZU" in wanted:
            zu_res = translator.translate_attack(base_text, "zu")
            if zu_res is None:
                logging.error(f"Skipping ZU variant for base attack {base_name} (translation failed).")
            else:
                meta_zu: Dict[str, Any] = {
                    "base_attack_name": base_name,
                    "variant_type": "mono",
                    "source_language": "en",
                    "target_language": "zu",
                    "target_languages": None,
                    "language_pair": None,
                    "split_method": None,
                    "translation_model": zu_res.translation_model,
                    "translation_prompt_version": zu_res.translation_prompt_version,
                    "variant_label": "ZU",
                }
                expanded.append(
                    AttPrompt(
                        att_prompt=zu_res.text,
                        category=atk.category,
                        name=base_name,
                        metadata=meta_zu,
                    )
                )

        # Mixed BN+ZU
        if "BN+ZU" in wanted:
            seg1, seg2, split_method = split_attack_text(base_text)
            if not seg1 or not seg2:
                logging.error(
                    f"Skipping mixed BN+ZU variant for {base_name} (split produced empty segment)."
                )
            else:
                mixed_res = translator.translate_mixed_bn_zu(base_text, seg1, seg2, split_method)
                if mixed_res is None:
                    logging.error(
                        f"Skipping mixed BN+ZU variant for base attack {base_name} (translation failed)."
                    )
                else:
                    meta_mixed: Dict[str, Any] = {
                        "base_attack_name": base_name,
                        "variant_type": "mixed",
                        "source_language": "en",
                        "target_language": None,
                        "target_languages": ["bn", "zu"],
                        "language_pair": "bn+zu",
                        "split_method": split_method,
                        "translation_model": mixed_res.translation_model,
                        "translation_prompt_version": mixed_res.translation_prompt_version,
                        "variant_label": "BN+ZU",
                    }
                    expanded.append(
                        AttPrompt(
                            att_prompt=mixed_res.text,
                            category=atk.category,
                            name=base_name,
                            metadata=meta_mixed,
                        )
                    )

    # Ensure labels exist, then append English-output instructions (all variants).
    for atk in expanded:
        meta = atk.get_metadata()
        if "variant_label" not in meta:
            meta["variant_label"] = _variant_label(meta)
        label = meta["variant_label"]
        atk.att_prompt = append_english_output_instruction(atk.att_prompt, label)

    return expanded

