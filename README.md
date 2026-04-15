# Polyglot-Raccoon

Polyglot-Raccoon is a multilingual extension of the Raccoon prompt-extraction benchmark. It studies prompt extraction against hidden application-level system prompts in LLM-integrated applications by adapting a selected subset of Raccoon attacks into multilingual variants and measuring how attack language changes prompt extraction success and behavior.

This repository is built on top of the original Raccoon benchmark and codebase. Polyglot-Raccoon is a scoped multilingual extension with additional features, not a benchmark built from scratch.

## What this repository adds

- Multilingual attack expansion on top of the Raccoon attack pipeline
- Four attack language variants: `EN`, `BN`, `ZU`, `BN+ZU`
- Translate-to-English defense condition (`--translate_attack_to_english`)
- System-security-suffix defense condition (`--append_system_security_suffix`)
- Multilingual result summarization with semantic leakage metrics

## Installation


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r linux_requirements.txt
```

## Dataset layout

- Prompt folders are under `Data/GPTs50/` and `Data/GPTs146/`.
- Each GPT/app subdirectory contains `system_prompt.md`.
- Attack sets are under `Data/attacks/` (for example `singular_attacks_deflesstop5`).

## Attack variants

- `EN`: original English attack
- `BN`: Bengali translation
- `ZU`: Zulu translation
- `BN+ZU`: deterministic mixed-language attack

`BN+ZU` is constructed by splitting the original English attack into two parts, translating one part to Bengali and the other to Zulu, then concatenating in original order to preserve attack intent.

## Defense conditions

- **undefended**: attack is submitted directly
- **translate-to-English**: attacker query is translated to English before submission to the victim model
- **system-security-suffix**: a fixed security suffix is appended to the application-level system prompt before evaluation

## LLM Models (`--model_name`)

The following named model options are defined in `run_raccoon_gang.py`:

- `gpt-4`
- `gpt-3.5`
- `gpt-3.5-0613`
- `gpt-3.5-0125`
- `gpt-5.4-nano`
- `gpt-5.4-nano-2026-03-17`
- `gemini-pro`
- `llama2_chat_70B`
- `mixtral_8x7B`
- `llama3.1_8b_openrouter`
- `mixtral_8x7b_openrouter`
- `llama3.3_70b_instruct_openrouter`
- `gpt-5.4-nano_openrouter`

You can also pass a direct provider model id as `--model_name`; unknown names are used as is.

## `.env` API key setup

Create a root `.env` file with the following keys:

```bash
OPENAI_API_KEY="your-openai-key"
OPENROUTER_API_KEY="your-openrouter-key"
```

Notes:

- `OPENROUTER_API_KEY` is required when running victim models through `--provider openrouter`.
- `OPENAI_API_KEY` is used for semantic embeddings (`--enable_semantic_chunk_leakage`) unless `RACCOON_SEMANTIC_EMBEDDING_API_KEY` is set.
- `run_raccoon_gang.py` loads `.env` automatically if `python-dotenv` is installed (included in `linux_requirements.txt`).

## Main commands

Run commands from the repository root.

### Run multilingual benchmark

```bash
# replace --model_name with any model
python run_raccoon_gang.py \
  --provider openrouter \
  --model_name llama3.3_70b_instruct_openrouter \
  --enable_multilingual_attacks \
  --enable_semantic_chunk_leakage \
  --gpts_path "./Data/GPTs146" \
  --attack_path "./Data/attacks/singular_attacks_deflesstop5" \
  --ref_def_path "./Data/reference/gpts196_defense_prompt.json" \
  --def_tmpl_path "./Data/defenses/defense_template.json" \
  --use_sys_template \
  --use_defenseless_user_prompt
```

### Summarize multilingual results

```bash
python scripts/summarize_multilingual_results.py \
  --results_dir "results/[RUN_NAME]" \
  --semantic
```

### Run all three defense modes

```bash
# replace --model_name with any model
python run_raccoon_gang.py \
  --provider openrouter \
  --model_name llama3.3_70b_instruct_openrouter \
  --run_all_three_benchmark_modes \
  --enable_multilingual_attacks \
  --enable_semantic_chunk_leakage \
  --gpts_path "./Data/GPTs146" \
  --attack_path "./Data/attacks/singular_attacks_deflesstop5" \
  --ref_def_path "./Data/reference/gpts196_defense_prompt.json" \
  --def_tmpl_path "./Data/defenses/defense_template.json" \
  --use_sys_template \
  --use_defenseless_user_prompt
```

## Multilingual options

The following options can be used:

- `--enable_multilingual_attacks`: enable EN/BN/ZU/BN+ZU attack expansion
- `--multilingual_variants`: select subset (`en,bn,zu,bn+zu`)
- `--enable_semantic_chunk_leakage`: enable semantic leakage analysis during benchmark runs
- `--translate_attack_to_english`: enable translate-to-English defense mode
- `--append_system_security_suffix`: enable system-security-suffix defense mode
- `--run_all_three_benchmark_modes`: run undefended + translate-to-English + system-security-suffix in one invocation

Useful related options:

- `--translation_provider`, `--translation_model`
- `--attack_to_english_provider`, `--attack_to_english_model`

## Results

Results are written under `results/` (for example `results/run_YYYYMMDD_HHMMSS/`). Three benchmark results for GPT 5.4-nano, GPT 3.5-turbo-0125, and Llama 3.3 70B Instruct have been included in this repository.

`scripts/summarize_multilingual_results.py --semantic` can report:

- mean ASR by defense/mode condition and language variant
- semantic candidate rate
- mean true semantic score
- mean semantic margin

## Acknowledgment

Polyglot-Raccoon is built on top of the original Raccoon benchmark and codebase. 

Wang, Junlin & Yang, Tianyi & Xie, Roy & Dhingra, Bhuwan. (2024). Raccoon: Prompt Extraction Benchmark of LLM-Integrated Applications. 13349-13365. 10.18653/v1/2024.findings-acl.791. 

