# Multi-turn Memory Degradation Experiment

This package is a complete first-pass pipeline for the experiment:

> A user reveals information across several chat turns. Later, the model is asked about that earlier information. We measure whether the model answers correctly, and whether its final answer step actually attends back to the earlier relevant user span.

## What this package does

1. **Builds a dataset** from chat conversations plus an inserted controllable fact.
2. **Runs four main experiment settings** (plus two experimental ones):
   - `packed_single_turn`
   - `multi_turn_neutral`
   - `multi_turn_dataset`
   - `multi_turn_self_generated`
3. **Stores both predictions and attention-based retrieval metrics**.

## Core idea

- The **haystack** is a natural chat conversation.
- The **needle** is a short fact inserted into one user turn.
- The final user turn asks about that earlier fact.
- We compare whether the model retrieves that fact under different conversation conditions.

## Why these settings exist

### Core settings (default)

- `packed_single_turn`: all user information is packed into one prompt. This is the non-multi-turn baseline.
- `multi_turn_neutral`: multi-turn structure exists, but assistant replies are fixed neutral text.
- `multi_turn_dataset`: multi-turn structure uses assistant replies already in the dataset.
- `multi_turn_self_generated`: the evaluated model generates the intermediate assistant replies itself.

### Experimental settings (opt-in)

- `recap_final_turn`: same as `multi_turn_dataset`, but a recap of all user-provided information is appended just before the final question.
- `snowball_user`: same as `multi_turn_dataset`, but each user turn re-states all previous user utterances.

Pass experimental settings via `--settings`:

```bash
python run_experiment.py \
  --settings packed_single_turn recap_final_turn snowball_user \
  ...
```

This lets you separate:
- pure turn-distance / interference effects,
- natural-conversation effects,
- self-generated context effects,
- explicit repetition / recap effects.

## Files

- `generate_dataset.py`: creates experiment JSONL from chat data.
- `run_experiment.py`: runs the model and writes result JSONL files.
- `aggregate_results.py`: creates a compact CSV summary.
- `sample_schema.json`: reference format for each dataset sample.
- `requirements.txt`: suggested packages.
- `src/`: loaders, builders, model wrapper, metrics, runner.

## Install

```bash
pip install -r requirements.txt
```

## Build dataset

### Option 1: OpenAssistant / OASST1

```bash
python generate_dataset.py \
  --chat_source oasst1 \
  --oasst_split train \
  --output_jsonl data/oasst_memory_probe.jsonl \
  --num_samples 500 \
  --min_user_turns 4 \
  --min_post_fact_user_turns 2 \
  --max_total_turns 10
```

### Option 2: local ShareGPT-style JSON

```bash
python generate_dataset.py \
  --chat_source local_sharegpt \
  --sharegpt_path /path/to/sharegpt_like.json \
  --output_jsonl data/sharegpt_memory_probe.jsonl \
  --num_samples 500
```

## Run experiment

### H100-fast recommended run

```bash
python run_experiment.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --input_jsonl data/oasst_memory_probe.jsonl \
  --output_dir outputs/qwen25_7b \
  --settings packed_single_turn multi_turn_neutral multi_turn_dataset multi_turn_self_generated \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --max_new_tokens 16 \
  --capture_last_k_steps 1 \
  --assistant_temperature 0 \
  --answer_temperature 0
```

> **Note on `--attn_implementation`:** `output_attentions=True` is **not supported** by `flash_attention_2`. If you need attention capture, use `--attn_implementation eager` or `--attn_implementation sdpa`. Flash Attention is fine for generation-only runs without attention analysis.


## Test features

### 1) Offline package tests

Runs the built-in non-GPU sanity checks for settings, dataset building, and fact-span recovery:

```bash
python run_tests.py
```

### 2) Dataset/config self-test

Validates a generated JSONL file against the selected settings without loading a model:

```bash
python run_experiment.py   --input_jsonl data/oasst1_retrieval_style.jsonl   --output_dir outputs/self_test   --settings packed_single_turn multi_turn_neutral multi_turn_dataset multi_turn_self_generated   --self_test   --self_test_num_samples 10
```

This checks that:
- messages can be built for every requested setting,
- the final message is a user question,
- and fact spans can be recovered for the inspected samples.

## Aggregate results

```bash
python aggregate_results.py \
  --summary_jsonl outputs/qwen25_7b/summary.jsonl \
  --output_csv outputs/qwen25_7b/summary_grouped.csv
```

## Output files

### `summary.jsonl`
One row per `(sample, setting)`.
Contains:
- sample id
- setting
- prediction
- gold answer
- correctness
- mean target-fact attention mass
- top-k hit rate

### `details.jsonl`
One row per `(sample, setting)` with rich debugging info.
Contains:
- rendered prompt text
- full messages used in that run
- final prediction
- fact token spans (as `[start, end]` lists)
- per-layer/head target attention mass
- top attended token indices

### `summary_grouped.csv`
Grouped averages by setting.
Contains mean accuracy and attention metrics.

## Practical notes

- Start with `multi_turn_neutral` and `multi_turn_self_generated`.
- Keep `max_new_tokens` short at first.
- Use `capture_last_k_steps=1` for quick experiments; larger values capture attention at multiple generated-token positions.
- For debugging, start with a 3B or 7B model before scaling up.
- Use `--attn_implementation eager` when you need attention maps.


## Updated needle generation

The current dataset builder now follows a retrieval-head / needle-in-a-haystack style more closely:
- the haystack still comes from a natural chat dataset,
- but the memory target is now a short procedural synthetic needle,
- with a fixed direct retrieval question,
- and no distractor injection.

Example needles:
- `The special magic Seoul number is 48173.`
- `The secret code for Paris is 10482.`
- `The reference ID for Toronto is 77120.`

Example run:

```bash
python generate_dataset.py   --chat_source oasst1   --oasst_split train   --output_jsonl data/oasst1_retrieval_style.jsonl   --num_samples 1000   --needle_style mixed
```
