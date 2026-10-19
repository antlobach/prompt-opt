# Prompt optimization for TypeSafe Jev

This repository uses [GEPA](https://github.com/gepa-ai/gepa) to optimize the question and ten-level rubric passed to TypeSafe Jev's `Score` API. The experiment predicts WineMag critic scores from tasting notes and compares the optimized prompt with a fixed baseline on an 800-row holdout set.

GEPA changes prompt text only. It does not fine-tune Jev or modify model weights.

## Experiment design

The notebook runs the full experiment:

1. Download a pinned WineMag CSV and select 2,000 unique tasting notes with a seeded shuffle.
2. Reserve 1,200 rows for development and 800 rows for the final test.
3. Split the development rows into 960 GEPA training rows and 240 validation rows.
4. Ask Jev for a baseline score over ten ordered quality levels.
5. Let GEPA propose new `instructions` and `rubric` values, using an OpenRouter model for reflection.
6. Select the best candidate on the validation rows.
7. Evaluate the selected candidate once on the untouched test rows using RMSE and Spearman rank correlation.

Each candidate must contain one question ending in `?` and a JSON array of ten distinct quality levels ordered from worst to best. Candidate evaluation uses negative squared error, which GEPA maximizes.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenRouter API key for GEPA's reflection model
- A TypeSafe API key for Jev evaluations

The experiment makes thousands of model requests at its default settings. Check both providers' pricing and rate limits before running it.

## Setup

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/antlobach/prompt-opt.git
cd prompt-opt
uv sync --locked
```

Create `.env` in the repository root:

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
TYPESAFE_API_KEY=your-typesafe-key
```

You can also choose another OpenRouter model:

```dotenv
OPENROUTER_MODEL=openai/gpt-5.6-sol
```

The notebook defaults to `openai/gpt-5.6-sol` for reflection and `jev-latest` for evaluation.

## Run the notebook

Start JupyterLab from the repository root:

```bash
uv run jupyter lab notebooks/jev_prompt_tuning.ipynb
```

Run the cells from top to bottom. The first cell controls whether GEPA starts a new search:

```python
START_FRESH = True
```

Set it to `False` before rerunning the notebook if you want GEPA to resume from `gepa_jev_run/`. TypeSafe responses remain cached in `json_cache.json` in either mode.

To reduce the default 100 GEPA proposal iterations for a trial run, set `GEPA_ITERATIONS` before starting JupyterLab:

```bash
GEPA_ITERATIONS=3 uv run jupyter lab notebooks/jev_prompt_tuning.ipynb
```

Lowering the iteration count reduces GEPA proposals, but the notebook still evaluates the baseline across all 2,000 rows.

## Configuration

| Name | Required | Default | Purpose |
|---|---:|---|---|
| `OPENROUTER_API_KEY` | Yes | None | Authenticates the GEPA reflection model |
| `TYPESAFE_API_KEY` | Yes | None | Authenticates Jev scoring requests |
| `OPENROUTER_MODEL` | No | `openai/gpt-5.6-sol` | Selects the reflection model through OpenRouter |
| `GEPA_ITERATIONS` | No | `100` | Sets the maximum number of GEPA candidate proposals |
| `START_FRESH` | Notebook setting | `True` | Deletes GEPA search state before optimization |

The evaluator uses eight worker threads. TypeSafe requests use a 120-second timeout and retry transient connection, timeout, server, and rate-limit errors.

## Generated files

The notebook writes runtime state and results to the repository root:

| Path | Contents |
|---|---|
| `json_cache.json` | Cached dataset and TypeSafe responses |
| `gepa_jev_run/` | GEPA checkpoints used for resume |
| `optimized_jev_candidate.json` | Winning question and ten-level rubric |
| `gepa_jev_result.json` | Full serialized GEPA result |

Git ignores these files because they are generated locally. Keep `json_cache.json` if you want repeated runs to reuse paid TypeSafe responses.

## Reproducibility notes

- The WineMag source URL includes a fixed dataset revision.
- Sampling uses seed `0` after duplicate tasting notes are removed.
- The 800 test labels do not enter GEPA training or validation.
- GEPA candidates reuse a calibration offset learned from the training portion of the development set.
- The final comparison recalibrates each prompt on all 1,200 development rows, then computes RMSE and Spearman correlation on the holdout once.

Provider model aliases can change upstream. Record the resolved provider models and generated result files when comparing runs made on different dates.

## Repository layout

```text
.
├── notebooks/
│   └── jev_prompt_tuning.ipynb  # Complete experiment
├── pyproject.toml                # Python requirements
└── uv.lock                       # Locked dependency graph
```

## License

Released under the [MIT License](LICENSE).
