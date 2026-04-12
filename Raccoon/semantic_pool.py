"""Build the pool of hidden prompts for semantic_metric_v2 negative controls."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from Raccoon.prompt import SysPrompt


def build_semantic_prompt_pool(
    gpts_paths: Iterable[Path],
    ref_defenses: dict,
    *,
    use_original_user_prompt: bool,
    use_defenseless_user_prompt: bool,
    use_original_user_defenses: bool,
    use_custom_defenses: bool,
    custom_defense: str,
    defense_position: str = "BOT",
) -> List[str]:
    """
    One hidden `user_sys_prompt` string per GPT folder, matching benchmark_single logic
    (same string stored as `prompt` in results).
    """
    pool: List[str] = []
    for gpts in gpts_paths:
        sp = SysPrompt(ref_defenses)
        sp.load_gpts(Path(gpts))
        s = sp.get_system_prompt(
            use_original_user_prompt,
            use_defenseless_user_prompt,
            use_original_user_defenses,
            use_custom_defenses,
            custom_defense,
            defense_position,
        )
        pool.append(s)
    return pool
