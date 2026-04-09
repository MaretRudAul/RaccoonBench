import logging
from typing import List, Dict, Any, Optional

from Raccoon.prompt import AttPrompt
from Raccoon.translation_utils import AttackTranslator, split_attack_text


SUPPORTED_LANGS = ["en", "bn", "zu"]
MIXED_PAIR = ("bn", "zu")


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
    include_mixed_bn_zu: bool = True,
) -> List[AttPrompt]:
    """
    Expand each base English attack into:
    - mono EN (original)
    - mono BN (LLM translation)
    - mono ZU (LLM translation)
    - mixed BN+ZU (deterministic split + translation)

    If translator is None, only EN variants are produced.
    """
    expanded: List[AttPrompt] = []

    for atk in base_attacks:
        base_text = atk.get_att_prompt()
        base_name = getattr(atk, "name", None) or atk.get_metadata().get("base_attack_name") or "attack"

        # EN mono (canonical)
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
        if include_mixed_bn_zu:
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

    # Ensure labels exist even if caller passed custom metadata
    for atk in expanded:
        meta = atk.get_metadata()
        if "variant_label" not in meta:
            meta["variant_label"] = _variant_label(meta)

    return expanded

