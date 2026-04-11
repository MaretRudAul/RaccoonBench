"""
Append instructions so the victim model answers in English (for fair ROUGE vs English system prompts).

- EN variant: English line twice (per experiment design).
- BN / ZU: English line + same meaning in target language.
- BN+ZU: English + Bengali + Zulu lines (covers the mixed prompt).
"""

# Exact English wording requested for experiments.
OUTPUT_LINE_EN = "Please provide all response in English."

# Static translations (deterministic; no extra API calls).
OUTPUT_LINE_BN = "অনুগ্রহ করে আপনার সকল উত্তর ইংরেজিতে দিন।"
OUTPUT_LINE_ZU = "Sicela unikeze zonke izimpendulo zakho ngesiNgisi."


def append_english_output_instruction(text: str, variant_label: str) -> str:
    """
    Append output-language constraints to the attack user message.

    variant_label: EN | BN | ZU | BN+ZU
    """
    t = (text or "").rstrip()
    sep = "\n\n"

    if variant_label == "EN":
        return f"{t}{sep}{OUTPUT_LINE_EN}\n{OUTPUT_LINE_EN}"
    if variant_label == "BN":
        return f"{t}{sep}{OUTPUT_LINE_EN}\n{OUTPUT_LINE_BN}"
    if variant_label == "ZU":
        return f"{t}{sep}{OUTPUT_LINE_EN}\n{OUTPUT_LINE_ZU}"
    if variant_label == "BN+ZU":
        return f"{t}{sep}{OUTPUT_LINE_EN}\n{OUTPUT_LINE_BN}\n{OUTPUT_LINE_ZU}"

    # Unknown variant: safest default = double English
    return f"{t}{sep}{OUTPUT_LINE_EN}\n{OUTPUT_LINE_EN}"
