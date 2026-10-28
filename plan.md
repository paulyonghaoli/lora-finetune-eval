LoRA Fine-Tuning Evaluation Lab

A serious, reproducible framework for answering:

Did LoRA fine-tuning actually improve the model?
Where did it improve?
Where did it regress?
Was the improvement worth the cost?
Did it overfit?
Did it hurt general behavior?
Can we compare LoRA configs rigorously?

The current repo already has a solid foundation: it prepares a Dolly-15k subset, fine-tunes gpt2 with LoRA/PEFT, evaluates base vs fine-tuned models, logs to MLflow, and saves artifacts. It uses Hugging Face Transformers, PEFT, datasets, MLflow, ROUGE, NumPy, and Torch.

A few important technical issues to address:

The README says LoRA is applied to attention projections, but target_modules=["q_proj", "v_proj"] may not match GPT-2 module names; GPT-2 typically uses modules like c_attn, so current LoRA targeting may fail or train nothing depending on PEFT behavior.
The training code labels the full prompt + response, so the model is trained to predict the instruction text too. For instruction tuning, you usually want loss only on the response portion.
Evaluation currently computes useful metrics, but it should export metrics.json, log eval metrics to MLflow, and support richer pre/post regression analysis.
Roadmap
Phase 1 — Make the experiment correct

Add:

config file
model-family-aware LoRA target modules
response-only loss masking
train/eval reproducibility metadata
metrics JSON
MLflow logging for evaluation
tests for prompt formatting and label masking
Phase 2 — Make evaluation serious

Add:

bootstrap confidence intervals
per-category breakdown
win/loss/tie analysis
hallucination / format / verbosity regressions
baseline vs LoRA comparison report
failure clustering
Phase 3 — Compare LoRA strategies

Add experiment grid:

rank r
alpha
dropout
target modules
learning rate
epochs
response-only vs full causal loss
full fine-tune vs LoRA if small enough
Phase 4 — Modernize models

Add small modern instruct models:

Qwen small instruct
TinyLlama
Phi small models
Mistral/Qwen adapters if hardware allows
Phase 5 — Connect to your larger portfolio

Use this repo to train/evaluate:

a verifier
a router
a skill matcher
an instruction-following adapter
an agent trace summarizer

That connects directly to your agentic AI path and your enterprise LLM work. Your career materials already emphasize LLM planning, vector search, embeddings, and tool-use pipelines , so this repo should become the “fine-tuning and evaluation” leg of that portfolio.

Claude Code prompt
You are working in my repo:
paulyonghaoli/lora-finetune-eval

Long-term goal:
Transform this repo from a simple LoRA fine-tuning demo into “LoRA Fine-Tuning Evaluation Lab” — a reproducible framework for testing whether parameter-efficient fine-tuning actually improves an LLM, where it regresses, whether it overfits, and which LoRA configuration is worth using.

Do NOT rewrite everything at once.
Build a stable, correct foundation first.

Current repo:
- data/prepare.py prepares Dolly-15k closed_qa examples
- training/train.py fine-tunes gpt2 with LoRA/PEFT
- evaluation/evaluate.py compares base vs fine-tuned model
- evaluation/error_analysis.py clusters failures
- main.py supports prepare/train/evaluate/error-analysis/all
- MLflow is used for training logs
- results include eval outputs/report

==================================================
PHASE 1 TARGET — CORRECT + REPRODUCIBLE LORA EVAL
==================================================

Main objectives:
1. Fix model-family-aware LoRA target modules
2. Add response-only loss masking
3. Add config-driven training/evaluation
4. Add richer result exports
5. Add evaluation MLflow logging
6. Add statistical comparison
7. Add tests and CI
8. Keep default demo runnable on CPU/small GPU

==================================================
PART 1 — CONFIG FILE
==================================================

Add config.yaml.

Example:

seed: 42

dataset:
  name: databricks/databricks-dolly-15k
  category: closed_qa
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  max_examples: 1000

model:
  name: gpt2
  model_family: gpt2
  device: auto
  max_length: 512

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  bias: none
  target_modules: auto

training:
  epochs: 3
  batch_size: 8
  learning_rate: 2e-4
  warmup_steps: 100
  weight_decay: 0.01
  eval_steps: 200
  save_steps: 200
  gradient_accumulation_steps: 1
  response_only_loss: true

generation:
  max_new_tokens: 128
  do_sample: false
  temperature: 0.0

paths:
  data_dir: data
  checkpoint_dir: checkpoints
  results_dir: results
  mlflow_dir: mlruns

Update main.py:
python main.py prepare --config config.yaml
python main.py train --config config.yaml
python main.py evaluate --config config.yaml
python main.py all --config config.yaml

Preserve current CLI defaults.

==================================================
PART 2 — MODEL-FAMILY-AWARE LORA TARGETS
==================================================

Current training code uses:
target_modules=["q_proj", "v_proj"]

This may not work correctly for GPT-2.
Implement a helper:

training/lora_utils.py

Functions:
- infer_model_family(model_name: str) -> str
- get_lora_target_modules(model_family: str) -> list[str]
- validate_lora_targets(model, target_modules) -> list[str]

Suggested defaults:
- gpt2: ["c_attn"]
- llama/mistral/qwen-like: ["q_proj", "v_proj"]
- t5/flan-t5 if supported later: document separately
- fallback: inspect named_modules and warn clearly

Requirements:
- If no target modules match, raise a clear ValueError.
- Log matched module names.
- Save LoRA config to checkpoints/lora_config.json.

Add tests using fake module names or tiny mock model.

==================================================
PART 3 — RESPONSE-ONLY LOSS MASKING
==================================================

Current tokenization labels the full prompt + output.
For instruction tuning, implement response-only labels by setting prompt tokens to -100.

Update _tokenize_batch.

Requirements:
- Build prompt without output:
  ### Instruction:
  ...
  ### Input:
  ...
  ### Response:

- Build full text:
  prompt + output

- Tokenize full text.
- Tokenize prompt alone to find prompt length.
- Set labels[:prompt_length] = -100.
- Keep labels for response tokens.
- Respect max_length.
- Add config flag:
  response_only_loss: true/false

Add tests:
- prompt tokens are masked
- response tokens are not masked
- examples with and without input
- truncation does not crash

==================================================
PART 4 — DATA PREPARATION IMPROVEMENTS
==================================================

Improve data/prepare.py.

Add:
- max_examples support
- minimum instruction/output length filters
- duplicate removal
- deterministic split
- dataset_summary.json

dataset_summary.json should include:
- dataset name
- category
- total raw examples
- filtered examples
- train/val/test counts
- average instruction length
- average input length
- average output length
- category distribution
- seed

Add tests with small in-memory dataset.

==================================================
PART 5 — TRAINING METADATA
==================================================

Save:
checkpoints/training_metadata.json

Include:
- timestamp
- git commit if available
- Python version
- torch version
- transformers version
- peft version
- model name
- LoRA config
- trainable parameter count
- total parameter count
- trainable percentage
- dataset summary
- seed
- device

Log the same key fields to MLflow.

==================================================
PART 6 — EVALUATION OUTPUTS
==================================================

Currently evaluation writes eval_outputs.jsonl and eval_report.md.
Add:
results/metrics.json
results/per_example_metrics.jsonl
results/comparison_summary.json
results/eval_report.md

metrics.json should include:
- base_perplexity
- fine_perplexity
- delta_perplexity
- base_rouge_l
- fine_rouge_l
- delta_rouge_l
- base_exact_match
- fine_exact_match
- delta_exact_match
- base_format_adherence
- fine_format_adherence
- delta_format_adherence
- base_mean_length
- fine_mean_length
- delta_mean_length

per_example_metrics.jsonl should include:
- instruction
- input
- reference
- base_prediction
- fine_prediction
- base metrics
- fine metrics
- delta metrics
- winner: base/fine/tie
- failure_tags

comparison_summary.json should include:
- fine_wins
- base_wins
- ties
- regression_count
- improvement_count
- biggest_improvements
- biggest_regressions

==================================================
PART 7 — STATISTICAL ANALYSIS
==================================================

Create evaluation/stats.py.

Add:
- bootstrap confidence intervals for mean ROUGE-L
- bootstrap CI for perplexity delta
- paired win/loss/tie counts
- simple paired sign test if easy
- effect size summary

Keep dependency-free if possible.

Add report section:
“Are improvements statistically meaningful?”

Do not overclaim.

==================================================
PART 8 — ERROR ANALYSIS IMPROVEMENT
==================================================

Improve evaluation/error_analysis.py.

Add failure tags:
- empty_output
- too_short
- too_long
- copied_prompt
- format_violation
- low_rouge
- high_perplexity
- hallucination_risk_keyword_mismatch
- regression_vs_base

Cluster or group by these tags.

Output:
results/error_analysis.json
results/error_analysis.md

==================================================
PART 9 — MLflow EVALUATION LOGGING
==================================================

Currently MLflow is used in training.
Add eval logging:
- metrics.json metrics
- artifacts:
  - eval_report.md
  - metrics.json
  - comparison_summary.json
  - error_analysis.md if available

Use same experiment name:
lora-finetune-eval

Tag:
stage=evaluation
model_name
adapter_path
dataset_category
seed

==================================================
PART 10 — MODEL/BASELINE MODES
==================================================

Add modes:
- evaluate-base-only
- evaluate-adapter
- evaluate-merged-adapter if implemented later

Optional:
Add adapter merge command:
python main.py merge-adapter

If too much, document as TODO.

==================================================
PART 11 — GENERATION CONTROLS
==================================================

Make generation config-driven:
- max_new_tokens
- do_sample
- temperature
- top_p
- repetition_penalty

Default should be deterministic:
do_sample=false

Save generation_config.json in results.

==================================================
PART 12 — EXPERIMENT GRID SKELETON
==================================================

Add scripts/run_grid.py.

Purpose:
Run multiple LoRA configs and compare them.

Grid fields:
- r
- alpha
- dropout
- learning_rate
- epochs
- target_modules

Do not run large grid by default.
Add a tiny example grid config:
configs/tiny_grid.yaml

Output:
results/grid_summary.json
results/grid_report.md

This can be a skeleton if full execution is too heavy.

==================================================
PART 13 — TESTS
==================================================

Add pytest tests for:
- config loading
- prompt formatting
- response-only label masking
- LoRA target module inference
- dataset filtering/splitting
- metric computation
- bootstrap CI shape
- per-example winner logic
- error tag assignment
- MLflow logging can be mocked
- CLI smoke tests if feasible

Use tiny fake datasets/models where possible.
Do not require downloading GPT-2 in unit tests.

==================================================
PART 14 — CI
==================================================

Add GitHub Actions:

.github/workflows/ci.yml

Run:
- pip install -r requirements.txt
- pip install pytest
- pytest -v

Do not run full training in CI.
Add optional smoke test with mocked model/tokenizer if feasible.

==================================================
PART 15 — README UPDATE
==================================================

Update README but keep it concise.

Add:
- positioning: LoRA Fine-Tuning Evaluation Lab
- what it evaluates
- why base vs fine-tuned comparison matters
- quick start
- config-based usage
- response-only loss explanation
- LoRA target-module warning by model family
- output files
- MLflow
- roadmap

Do not overclaim that LoRA improves the model.
Emphasize this repo tests whether it improves.

==================================================
PART 16 — DOCS
==================================================

Create docs/ROADMAP.md.

Roadmap:
v0: Dolly-15k + GPT-2 LoRA + basic eval
v1: config + correct target modules + response-only loss
v2: statistical eval + error analysis
v3: LoRA grid comparison
v4: QLoRA / bitsandbytes optional
v5: instruction-following judge
v6: train verifier/router/skill adapters for agent systems
v7: paper-ready fine-tuning eval benchmark

Create docs/REFERENCES.md.

Include annotated references/topics:
- LoRA
- QLoRA
- PEFT
- instruction tuning
- response-only supervised fine-tuning
- Dolly-15k
- ROUGE limitations
- perplexity vs generation quality
- MLflow experiment tracking
- adapter merging
- catastrophic forgetting / regression testing

For each:
- key idea
- what this repo borrows
- future work

Create docs/AGENT_INTEGRATION.md.

Explain how this repo can support my other agent repos:
- fine-tune a verifier
- fine-tune a query router
- fine-tune a failure tagger
- fine-tune a trace summarizer
- evaluate whether fine-tuning improves agent components

==================================================
ENGINEERING CONSTRAINTS
==================================================

- Keep default demo small.
- Avoid requiring a large GPU.
- Do not commit model weights.
- Add checkpoints/ and mlruns/ to .gitignore if missing.
- No secrets.
- Avoid huge generated artifacts.
- Unit tests should not download large models.
- Keep code readable and typed.
- Prefer small modules.
- Preserve current CLI behavior where possible.
- Do not claim fine-tuning is better unless metrics show it.
- Be explicit about ROUGE/perplexity limitations.

==================================================
FINAL COMMANDS TO RUN
==================================================

After implementation, run:

pytest -v

If environment supports model download/training, also run:

python main.py prepare --config config.yaml
python main.py train --config config.yaml
python main.py evaluate --config config.yaml
python main.py error-analysis --config config.yaml

If full training is too heavy, run available unit/smoke tests and explain what was not run.

Then summarize:
- files changed
- bugs fixed
- new architecture
- tests added
- commands run
- remaining TODOs

Most important improvement: fix correctness first. A LoRA repo is only impressive if it is honest about whether fine-tuning helped, where it regressed, and whether the experimental setup was valid.