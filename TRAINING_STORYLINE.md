# Training Storyline

This file reconstructs the training trail from git commit messages.
It is a narrative derived from commit history, not a ground-truth experiment tracker.

## Scope

- Period covered: March 24, 2026 to April 5, 2026
- Source: `git log` commit subjects
- Goal: show what was tried, in order, across data prep, model/training stability, H100/H200 execution workflows, and the later chat-SFT path

## High-Level Arc

The training trail moved through eight clear phases:

1. Stand up the repo and cluster workflow.
2. Make dataset preparation actually work at cluster scale.
3. Define the training plan: two-stage pretrain plus Rust fine-tune.
4. Stabilize the training stack around Flash Attention, dtype issues, and single-GPU execution.
5. Switch the base stack to Qwen 3.5 and build smoke/full-train flows.
6. Turn H100/H200 training into a resumable Slurm workflow with W&B continuity and time-limit recovery.
7. Extend the resume/recovery tooling to the Rust stage-transition path.
8. Add a Nemotron-based chat-SFT pipeline and harden it against real cluster failures.

## Storyline

### Phase 1: Bootstrapping the project and cluster path

What was tried:

- Set up Etude as a compact language model project for Rust code generation.
- Added contributor and Khoury cluster setup instructions early, which suggests the work was intended to run on shared GPU infrastructure from the start.
- Tightened the operational docs around available H100/H200 resources, session limits, tmux, Ghostty, and checkpoint-based workflows.

What this says about the trail:

- The first concern was not model quality yet. It was making sure the repo could be used repeatedly on the cluster without losing work when sessions expired.
- The training trail started as an infrastructure problem before it became an optimization problem.

Evidence:

- `5ea83ba` Initial commit: Etude — compact language model for Rust code generation
- `810c2e4` Add CONTRIBUTING.md with setup guide and Khoury cluster instructions
- `a860acc` Add GPU monitor dashboard link to contributing guide
- `94e2476` Update GPU session docs with H200/H100 availability and 2h default
- `808a028` Add instructions for Ghostty terminal compatibility in contributing guide
- `854711d` Add checkpoint-based training workflow and Ghostty tmux tip
- `26e2b4d` Add venv activation to cluster setup instructions

### Phase 2: Data pipeline pressure and storage/OOM fixes

What was tried:

- Expanded data preparation documentation and added dataset preparation scripts.
- Moved toward parquet-based handling and scratch storage instead of relying on home directory quota.
- Fixed OOMs during data prep by reducing working-set size.
- Switched from slower row-wise streaming to faster download methods where possible.
- Added a Hugging Face streaming fallback so training could still proceed without a fully downloaded local dataset.

What this says about the trail:

- The first real bottleneck was data ingestion, not GPU math.
- Several commits point to the same pain: large datasets were too heavy to prepare naively on cluster nodes.
- The repo evolved toward a more fault-tolerant pipeline: local parquet when available, streaming fallback when not.

Evidence:

- `cb52e32` Expand data preparation docs with binary and parquet paths
- `8913a49` Add Rust dataset preparation scripts and update binary data handling
- `9de9bbb` Fix OOM in data prep by loading 10 files per part instead of full dataset
- `2c979e1` Add HF cache setup to prepare.sh and data prep notes to contributing guide
- `132db8e` Add --output-dir support to data prep scripts for scratch storage
- `a1a8f08` Rewrite data prep docs to reflect correct parquet streaming pipeline
- `181a791` Fix dataset download to use OptimalScale/ClimbMix with JSONL-to-parquet conversion
- `5218c8a` Default base dir to /scratch on cluster to avoid home quota issues
- `a4d75b5` Use streaming mode for dataset download to avoid OOM on large JSONL files
- `cad1a29` Add tqdm progress bars to dataset download for visibility on long runs
- `50547d2` Use huggingface_hub fast download instead of slow row-by-row streaming
- `2b72111` Add HuggingFace streaming fallback for training without local dataset download

### Phase 3: Training plan formation and tokenization tooling

What was tried:

- Introduced a two-stage training plan: FineWeb-Edu pretrain followed by Rust fine-tune.
- Wrote down token budgets and pipeline docs for that plan.
- Added tokenizer visualization to inspect tokenization behavior instead of treating it as a black box.

What this says about the trail:

- By this point the project had shifted from “can I train at all?” to “what exact curriculum should I run?”
- The trail suggests a deliberate attempt to separate generic language competence from domain specialization.
- Tokenization was important enough to inspect visually, which usually means the team was trying to catch hidden quality losses early.

Evidence:

- `154e1f1` Add two-stage training pipeline: FineWeb-Edu pretrain + Rust fine-tune
- `16f51d5` Update README with two-stage training pipeline and token budgets
- `973594b` Add tokenizer visualization script (colored HTML output)
- `00cba31` Add tokenizer visualization instructions to CONTRIBUTING.md

### Phase 4: Core training stability fixes

What was tried:

- Removed noisy banner output from training logs.
- Fixed multiple training/eval correctness issues: token-byte/vocab mismatch, MTP shape mismatch, Conv1d dtype mismatch, rms_norm upcasting, and float32 leaking into Flash Attention.
- Migrated from Flash Attention 3 to Flash Attention 4.
- Added fp16 AMP support for older GPUs.
- Reduced `torch.compile` cost by disabling tracing on SDPA and documenting debug paths.

What this says about the trail:

- The middle of the trail was dominated by “make training numerically and operationally trustworthy.”
- The commits read like a cleanup wave after actual runs exposed failure modes.
- This was not one bug. It was a cluster of issues around dtype consistency, attention kernels, and evaluation paths.

Evidence:

- `7d769df` Remove ASCII banner from training output
- `9cf6f7b` Fix token_bytes vs vocab_size mismatch by padding token_bytes to model vocab size
- `27a77d4` Fix MTP loss shape mismatch when loss_reduction='none' (BPB eval)
- `557f5ab` Fix Conv1d dtype mismatch in GatedDeltaNet under bfloat16 AMP
- `bb126cf` Fix rms_norm upcasting to float32 before Flash Attention
- `8d5ae29` Guard against float32 reaching Flash Attention 3
- `5367fea` Migrate Flash Attention from 3 to 4 (CuTeDSL)
- `6add339` Replace kernels dependency with flash-attn-4
- `65fd7f2` Add fp16 AMP support for pre-Ampere GPUs
- `4ef541f` Add torch.compile debug docs and restore -- separator in SLURM script
- `1f584d4` Disable torch.compile tracing on SDPA to reduce compile time
- `5d6808e` Fix training and eval edge cases

### Phase 5: Single-GPU H100 execution becomes the main operating mode

What was tried:

- Added Slurm batch scripts and CUDA module loading for H100 training.
- Clarified single-GPU training flags and smaller training examples.
- Fixed two-stage checkpoint handoff.
- Documented the single-GPU and two-stage commands more explicitly.

What this says about the trail:

- The project converged on “single strong GPU, resumed repeatedly” rather than assuming long uninterrupted multi-GPU runs.
- A lot of the work here is operational hardening: make H100 runs repeatable, make stage transitions less brittle, and reduce ambiguity in command lines.

Evidence:

- `1ae5682` Add SLURM batch script and docs for H100 training
- `8586d6a` Load CUDA module in SLURM script to fix missing libcudnn
- `6ed6bae` Clarify single-GPU training flags in docs
- `68d944b` Fix twostage checkpoint handoff
- `af46467` Document twostage single-GPU command
- `520064c` Add smaller single-GPU training examples

### Phase 6: Architecture pivot to Qwen 3.5

What was tried:

- Switched base training and checkpoint loading to Qwen 3.5.
- Fixed a Qwen 3.5 DeltaNet dtype issue.
- Added configurable W&B logging.
- Added dedicated H100 smoke-run docs and checkpoint chat smoke flow.
- Documented single-H100 full-train commands.

What this says about the trail:

- This was a meaningful pivot, not a tiny refactor.
- The trail suggests the earlier stack was useful, but the project then re-centered around Qwen 3.5 as the practical base path.
- Smoke tests and chat-load docs appeared immediately, which implies the change was validated through quick checkpoint loops, not just code edits.

Evidence:

- `d04c379` Use Qwen 3.5 for base training and chat loading
- `e23e73f` Fix Qwen 3.5 DeltaNet training dtype
- `beedc17` Add configurable wandb logging and ignore uv.lock
- `9c50780` Document Qwen 3.5 H100 smoke run
- `094122f` Improve Qwen checkpoint chat smoke flow
- `41707ac` Clarify W&B setup for smoke runs
- `c89f7ea` Document single-H100 full-train commands

### Phase 7: Resume, observe, and survive cluster time limits

What was tried:

- Documented training log fields so runs could be interpreted consistently.
- Added W&B resume continuity across training restarts and later fixed continuity bugs.
- Added Slurm resume flows for both d24 H100 and H200.
- Added a watcher to follow and auto-resubmit time-limited Slurm jobs.
- Repeatedly refined the watcher to pick the newest log correctly, parse prefixes correctly, wait for logs to appear, and keep watching resumed chains.
- Logged every resumed step to W&B.
- Handled missing CORE eval bundles more gracefully.
- Fixed BOS handling in sampling.
- Tuned H100 resume defaults to be a bit faster and then documented those defaults.

What this says about the trail:

- The latest phase is clearly about long-running training under hostile scheduler constraints.
- The project stopped treating preemption/time limits as rare events and started treating them as the normal case.
- This is the strongest signal in the whole trail: the training loop became an operational system with resume semantics, observability, and recovery tooling.

Evidence:

- `a419a85` Document training log fields
- `5c4fef7` Resume W&B runs across training restarts
- `e3cb947` Document d24-h100 checkpoint chat command
- `c4826d3` Add Slurm resume flow for d24 H100
- `ce8c27d` Add H200 resume sbatch job
- `72ebebd` Clarify Slurm batch reattachment in README
- `884e4d9` Fix W&B resume continuity for base training
- `7499815` Add Slurm time-limit resubmission watcher
- `c44025d` Auto-detect newest Slurm log for watcher
- `aca9e6a` Use log mtime in Slurm watcher
- `b442034` Fix Slurm watcher prefix parsing
- `5d7d908` auto-watch slurm resume chain
- `e8aebcf` log every resumed step to wandb
- `5c21217` Wait for missing Slurm logs before watching
- `06bf073` Handle missing CORE eval bundle in training
- `f4b64c0` Fix BOS token handling in sampling
- `452e23b` Tune H100 resume training defaults
- `408735c` Document H100 resume launcher defaults

### Phase 8: Rust resume path and Nemotron chat-SFT become first-class

What was tried:

- Added a Rust stage-transition resume script and taught the Slurm watcher how to follow Rust runs too.
- Added a full Nemotron Cascade SFT Stage 2 preparation pipeline and a dedicated chat-SFT launcher.
- Repeatedly hardened Nemotron prep for real cluster usage: environment dependencies, import path issues, OOM reductions, compute-node launchers, H100 defaults, and narrower default subset selection.
- Shifted the default prep target toward the `instruction-following` subset, which suggests the chat-SFT path was narrowed from “all available chat data” to “the most directly useful subset.”
- Fixed multiple compatibility bugs uncovered only after the first real SFT launches: HuggingFace chat rendering, stale `token_bytes` cache mismatches, overlong conversation packing, and the fact that long jobs need periodic SFT checkpoints rather than end-of-run-only saves.

What this says about the trail:

- The project expanded from base-model pretrain and Rust specialization into a real chat fine-tuning branch.
- This phase reads like a live operational bring-up: create the new path, hit the actual cluster/runtime failures, then fix them one by one until the job can survive long enough to be useful.
- The final commits are especially telling: once SFT moved onto shared H100 time, periodic checkpointing and row-cap-aware packing became mandatory, not optional cleanup.

Evidence:

- `838d1e4` Add Rust stage-transition resume script
- `eb1cc1f` Auto-detect latest Rust stage-transition source step
- `ff8fd73` Add Rust support to Slurm time-limit watcher
- `a8de8e8` Default Rust resume launcher to checkpoint dir
- `92b909f` Add Nemotron chat SFT pipeline and Slurm job
- `d119e7c` Fix Nemotron prepare environment dependencies
- `fb98bea` Fix Nemotron prepare import path
- `b5ad25c` Reduce Nemotron prepare memory usage
- `ed553f1` Add Slurm launcher for Nemotron dataset prep
- `9c9c653` Switch Nemotron prep Slurm job to H100
- `8a60867` Default Nemotron prep to instruction-following and code
- `22b9b0b` Update chat SFT default checkpoint to step 8800
- `4d7324a` Harden Nemotron prep and default to instruction-following
- `e5ce18d` Add chat rendering to HuggingFace tokenizer
- `801a913` Refresh token byte cache for chat SFT
- `6f21095` Add periodic chat SFT checkpoints

## What You Have Tried, Condensed

If this needs to read like a short retrospective, the training trail looks like this:

- Tried to get the cluster workflow stable first: setup, tmux, checkpoints, scratch storage, and Slurm docs.
- Tried several dataset ingestion approaches until OOM and download speed issues were manageable: chunked loading, parquet conversion, fast downloads, and streaming fallback.
- Tried a two-stage training plan to separate general pretraining from Rust specialization.
- Tried to stabilize the actual training stack by fixing dtype bugs, eval bugs, token accounting issues, and attention kernel mismatches.
- Tried to modernize the kernel stack by moving to Flash Attention 4 and supporting different precision paths.
- Tried to make single-H100 training practical with better docs, examples, and Slurm launch scripts.
- Tried a model-stack pivot to Qwen 3.5 and then built smoke/full-train validation flows around it.
- Tried to make interrupted training survivable by adding W&B resume continuity, resume sbatch jobs, log watchers, auto-resubmission, and faster resume defaults.
- Tried to extend those resume semantics to the Rust stage-transition path instead of treating that phase as a one-off manual handoff.
- Tried to stand up a Nemotron-based chat-SFT branch, then hardened it through the exact failures that surfaced on cluster: dependency gaps, import-path issues, prep OOMs, tokenizer/rendering mismatches, stale cache mismatches, row-packing bugs, and missing periodic checkpoints.

## Raw Chronological Commit Trail

This appendix preserves the training-related commit trail in chronological order so the storyline above stays auditable.

```text
5ea83ba  2026-03-24  Initial commit: Etude — compact language model for Rust code generation
8913a49  2026-03-24  Add Rust dataset preparation scripts and update binary data handling
9de9bbb  2026-03-25  Fix OOM in data prep by loading 10 files per part instead of full dataset
2c979e1  2026-03-25  Add HF cache setup to prepare.sh and data prep notes to contributing guide
132db8e  2026-03-25  Add --output-dir support to data prep scripts for scratch storage
181a791  2026-03-25  Fix dataset download to use OptimalScale/ClimbMix with JSONL-to-parquet conversion
5218c8a  2026-03-25  Default base dir to /scratch on cluster to avoid home quota issues
a4d75b5  2026-03-25  Use streaming mode for dataset download to avoid OOM on large JSONL files
cad1a29  2026-03-25  Add tqdm progress bars to dataset download for visibility on long runs
50547d2  2026-03-25  Use huggingface_hub fast download instead of slow row-by-row streaming
2b72111  2026-03-25  Add HuggingFace streaming fallback for training without local dataset download
154e1f1  2026-03-25  Add two-stage training pipeline: FineWeb-Edu pretrain + Rust fine-tune
16f51d5  2026-03-25  Update README with two-stage training pipeline and token budgets
973594b  2026-03-25  Add tokenizer visualization script (colored HTML output)
7d769df  2026-03-25  Remove ASCII banner from training output
9cf6f7b  2026-03-25  Fix token_bytes vs vocab_size mismatch by padding token_bytes to model vocab size
27a77d4  2026-03-25  Fix MTP loss shape mismatch when loss_reduction='none' (BPB eval)
557f5ab  2026-03-25  Fix Conv1d dtype mismatch in GatedDeltaNet under bfloat16 AMP
bb126cf  2026-03-25  Fix rms_norm upcasting to float32 before Flash Attention
8d5ae29  2026-03-25  Guard against float32 reaching Flash Attention 3
5367fea  2026-03-26  Migrate Flash Attention from 3 to 4 (CuTeDSL)
65fd7f2  2026-03-26  Add fp16 AMP support for pre-Ampere GPUs
1ae5682  2026-03-26  Add SLURM batch script and docs for H100 training
8586d6a  2026-03-26  Load CUDA module in SLURM script to fix missing libcudnn
1f584d4  2026-03-26  Disable torch.compile tracing on SDPA to reduce compile time
5d6808e  2026-04-02  Fix training and eval edge cases
68d944b  2026-04-02  Fix twostage checkpoint handoff
520064c  2026-04-02  Add smaller single-GPU training examples
d04c379  2026-04-03  Use Qwen 3.5 for base training and chat loading
e23e73f  2026-04-03  Fix Qwen 3.5 DeltaNet training dtype
beedc17  2026-04-03  Add configurable wandb logging and ignore uv.lock
9c50780  2026-04-03  Document Qwen 3.5 H100 smoke run
094122f  2026-04-03  Improve Qwen checkpoint chat smoke flow
c89f7ea  2026-04-03  Document single-H100 full-train commands
a419a85  2026-04-04  Document training log fields
5c4fef7  2026-04-04  Resume W&B runs across training restarts
c4826d3  2026-04-04  Add Slurm resume flow for d24 H100
ce8c27d  2026-04-04  Add H200 resume sbatch job
884e4d9  2026-04-04  Fix W&B resume continuity for base training
7499815  2026-04-04  Add Slurm time-limit resubmission watcher
c44025d  2026-04-04  Auto-detect newest Slurm log for watcher
aca9e6a  2026-04-04  Use log mtime in Slurm watcher
b442034  2026-04-04  Fix Slurm watcher prefix parsing
5d7d908  2026-04-04  auto-watch slurm resume chain
e8aebcf  2026-04-04  log every resumed step to wandb
5c21217  2026-04-04  Wait for missing Slurm logs before watching
06bf073  2026-04-04  Handle missing CORE eval bundle in training
f4b64c0  2026-04-04  Fix BOS token handling in sampling
452e23b  2026-04-04  Tune H100 resume training defaults
408735c  2026-04-04  Document H100 resume launcher defaults
2b81189  2026-04-04  Add training storyline from commit history
838d1e4  2026-04-05  Add Rust stage-transition resume script
eb1cc1f  2026-04-05  Auto-detect latest Rust stage-transition source step
ff8fd73  2026-04-05  Add Rust support to Slurm time-limit watcher
a8de8e8  2026-04-05  Default Rust resume launcher to checkpoint dir
92b909f  2026-04-05  Add Nemotron chat SFT pipeline and Slurm job
d119e7c  2026-04-05  Fix Nemotron prepare environment dependencies
fb98bea  2026-04-05  Fix Nemotron prepare import path
b5ad25c  2026-04-05  Reduce Nemotron prepare memory usage
ed553f1  2026-04-05  Add Slurm launcher for Nemotron dataset prep
9c9c653  2026-04-05  Switch Nemotron prep Slurm job to H100
8a60867  2026-04-05  Default Nemotron prep to instruction-following and code
22b9b0b  2026-04-05  Update chat SFT default checkpoint to step 8800
4d7324a  2026-04-05  Harden Nemotron prep and default to instruction-following
e5ce18d  2026-04-05  Add chat rendering to HuggingFace tokenizer
801a913  2026-04-05  Refresh token byte cache for chat SFT
6f21095  2026-04-05  Add periodic chat SFT checkpoints
```
