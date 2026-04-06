"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import hashlib
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
from dataclasses import asdict
import wandb
import torch
from etude.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from etude.sft_dataloader import PackedConversationDataLoader
from etude.tokenizer import get_token_bytes
from etude.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state, build_model, load_checkpoint
from etude.loss_eval import evaluate_bpb
import torch.distributed as dist
from etude.flash_attention import HAS_FA4
from etude.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.nemotron_cascade_sft_stage2 import (
    NemotronCascadeSFTStage2,
    get_default_data_dir as get_default_chat_data_dir,
    has_prepared_data as has_prepared_chat_data,
)
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-model-tag", type=str, default=None, help="checkpoint directory name for SFT outputs (default: model-tag)")
parser.add_argument("--source-checkpoint-dir", type=str, default=None, help="explicit source checkpoint directory for fresh SFT starts")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume SFT from this step (-1 = disable)")
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
# Chat dataset
parser.add_argument(
    "--chat-dataset",
    type=str,
    default="auto",
    choices=["auto", "legacy", "nemotron-cascade-sft-stage-2"],
    help="Chat SFT dataset selection. 'auto' uses prepared Nemotron data if present, otherwise falls back to the legacy task mixture.",
)
parser.add_argument(
    "--chat-data-dir",
    type=str,
    default=None,
    help="Prepared chat dataset directory for Nemotron Stage 2 (default: ETUDE base dir under datasets/)",
)
parser.add_argument(
    "--chat-subsets",
    type=str,
    default=None,
    help="Optional comma-separated Nemotron subsets to use from prepared parquet shards",
)
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice)")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# Checkpoint paths and mode
base_dir = get_base_dir()
resuming = args.resume_from_step != -1
output_model_tag = args.output_model_tag or args.model_tag
if resuming and output_model_tag is None:
    raise ValueError("--resume-from-step requires --output-model-tag or --model-tag")

# Load the model and tokenizer
if resuming:
    model, tokenizer, meta = load_model(
        "sft",
        device,
        phase="train",
        model_tag=output_model_tag,
        step=args.resume_from_step,
    )
    checkpoint_source = "sft"
    checkpoint_step = args.resume_from_step
else:
    checkpoint_step = args.model_step
    if args.source_checkpoint_dir is not None:
        if checkpoint_step is None:
            raise ValueError("--source-checkpoint-dir requires --model-step")
        model, tokenizer, meta = build_model(
            args.source_checkpoint_dir,
            checkpoint_step,
            device,
            phase="train",
        )
    else:
        model, tokenizer, meta = load_model(
            "base",
            device,
            phase="train",
            model_tag=args.model_tag,
            step=checkpoint_step,
        )
    checkpoint_source = "base"

# Flash Attention status
if not HAS_FA4:
    print0("WARNING: Flash Attention 4 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run_id = None
wandb_run_id_from_checkpoint = False
if resuming:
    wandb_run_id = meta.get("wandb_run_id")
    wandb_run_id_from_checkpoint = wandb_run_id is not None
    if wandb_run_id is None:
        resume_tag = output_model_tag if output_model_tag is not None else "chat-sft"
        wandb_run_id = hashlib.sha1(f"{resume_tag}:{args.run}".encode("utf-8")).hexdigest()[:16]

if use_dummy_wandb:
    wandb_run = DummyWandb()
else:
    wandb_kwargs = dict(
        project=os.environ.get("WANDB_PROJECT", "etude-sft"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=args.run,
        config=user_config,
    )
    if resuming and wandb_run_id is not None:
        resume_mode = "must" if wandb_run_id_from_checkpoint else "allow"
        wandb_kwargs.update(id=wandb_run_id, resume=resume_mode)
        print0(f"Using W&B run id: {wandb_run_id}")
    wandb_run = wandb.init(**wandb_kwargs)
    wandb_run.define_metric("step")
    wandb_run.define_metric("*", step_metric="step")

orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
output_dirname = output_model_tag if output_model_tag else f"d{depth}"
user_config["resolved_output_model_tag"] = output_dirname
user_config["checkpoint_source"] = checkpoint_source
user_config["checkpoint_step"] = checkpoint_step
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(
    device=device,
    vocab_size=tokenizer.get_vocab_size(),
    tokenizer=tokenizer,
)

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
optimizer = model.setup_optimizer(embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# Restore optimizer state for resume, or optionally warm-start it from the source checkpoint.
if resuming:
    optimizer_data = load_optimizer_state("sft", device, rank=ddp_rank, model_tag=output_dirname, step=args.resume_from_step)
    if optimizer_data is None:
        raise FileNotFoundError(
            f"Missing SFT optimizer checkpoint for {output_dirname} step {args.resume_from_step}"
        )
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data
    for group in optimizer.param_groups:
        group["initial_lr"] = group.get("initial_lr", group["lr"])
    print0(f"Loaded optimizer state from SFT checkpoint step {args.resume_from_step}")
elif args.load_optimizer:
    if args.source_checkpoint_dir is not None:
        _, optimizer_data, _ = load_checkpoint(
            args.source_checkpoint_dir,
            args.model_step,
            device,
            load_optimizer=True,
            rank=ddp_rank,
        )
    else:
        optimizer_data = load_optimizer_state(
            "base",
            device,
            rank=ddp_rank,
            model_tag=args.model_tag,
            step=args.model_step,
        )
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

if not resuming:
    # Override the initial learning rate as a fraction of the base learning rate only for fresh starts.
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

def build_legacy_sft_datasets():
    identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    train_tasks = [
        SmolTalk(split="train"), # 460K rows of general conversations
        CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
        CustomJSON(filepath=identity_conversations_filepath), # 2 epochs of these
        *[MMLU(subset="auxiliary_train", split="train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
        SimpleSpelling(size=200000, split="train"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
        SpellingBee(size=80000, split="train"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
    ]
    train_dataset = TaskMixture(train_tasks)
    val_dataset = TaskMixture([
        SmolTalk(split="test"), # 24K rows in test set
        MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
        GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
    ]) # total: 24K + 14K + 1.32K ~= 39K rows
    print0(f"Training legacy SFT mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
    return train_dataset, val_dataset


def build_nemotron_sft_datasets(chat_data_dir):
    train_dataset = NemotronCascadeSFTStage2(
        split="train",
        data_dir=chat_data_dir,
        subsets=args.chat_subsets,
    )
    val_dataset = NemotronCascadeSFTStage2(
        split="val",
        data_dir=chat_data_dir,
        subsets=args.chat_subsets,
    )
    subset_suffix = f" | subsets: {args.chat_subsets}" if args.chat_subsets else ""
    print0(
        f"Training Nemotron Cascade SFT Stage 2: {len(train_dataset):,} train rows | "
        f"{len(val_dataset):,} val rows | data dir: {chat_data_dir}{subset_suffix}"
    )
    return train_dataset, val_dataset


chat_data_dir = args.chat_data_dir or get_default_chat_data_dir()
if args.chat_dataset == "legacy":
    resolved_chat_dataset = "legacy"
    train_dataset, val_dataset = build_legacy_sft_datasets()
elif has_prepared_chat_data(chat_data_dir):
    resolved_chat_dataset = "nemotron-cascade-sft-stage-2"
    train_dataset, val_dataset = build_nemotron_sft_datasets(chat_data_dir)
elif args.chat_dataset == "auto":
    resolved_chat_dataset = "legacy"
    print0(
        "Prepared Nemotron chat data not found; falling back to the legacy SFT mixture. "
        f"To prepare it, run: python data/nemotron-cascade-sft-stage-2/prepare.py --output-dir {chat_data_dir}"
    )
    train_dataset, val_dataset = build_legacy_sft_datasets()
else:
    raise FileNotFoundError(
        "Prepared Nemotron chat data not found. "
        f"Run: python data/nemotron-cascade-sft-stage-2/prepare.py --output-dir {chat_data_dir}"
    )
user_config["resolved_chat_dataset"] = resolved_chat_dataset
user_config["resolved_chat_data_dir"] = chat_data_dir

if resuming:
    saved_user_config = meta.get("user_config", {})
    saved_chat_dataset = saved_user_config.get("resolved_chat_dataset") or saved_user_config.get("chat_dataset")
    saved_chat_data_dir = saved_user_config.get("resolved_chat_data_dir")
    saved_chat_subsets = saved_user_config.get("chat_subsets")
    for name, current_value, saved_value in [
        ("chat-dataset", resolved_chat_dataset, saved_chat_dataset),
        ("chat-data-dir", os.path.abspath(chat_data_dir), os.path.abspath(saved_chat_data_dir) if saved_chat_data_dir else None),
        ("chat-subsets", args.chat_subsets, saved_chat_subsets),
        ("max-seq-len", args.max_seq_len, meta.get("max_seq_len")),
        ("device-batch-size", args.device_batch_size, meta.get("device_batch_size")),
        ("total-batch-size", args.total_batch_size, meta.get("total_batch_size")),
    ]:
        if saved_value is not None and current_value != saved_value:
            raise ValueError(
                f"SFT resume requires --{name}={saved_value}, but got {current_value}"
            )

resume_state_dict = meta.get("dataloader_state_dict") if resuming else None
train_loader = PackedConversationDataLoader(
    dataset=train_dataset,
    tokenizer=tokenizer,
    device_batch_size=args.device_batch_size,
    max_seq_len=args.max_seq_len,
    split="train",
    device=device,
    device_type=device_type,
    ddp_rank=ddp_rank,
    ddp_world_size=ddp_world_size,
    num_iterations=args.num_iterations,
    resume_state_dict=resume_state_dict,
)

def build_val_loader():
    val_loader = PackedConversationDataLoader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        device_batch_size=args.device_batch_size,
        max_seq_len=args.max_seq_len,
        split="val",
        device=device,
        device_type=device_type,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        num_iterations=args.num_iterations,
    )
    for inputs, targets, state_dict in val_loader:
        yield inputs, targets

progress = 0.0
checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)

# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
x, y, train_loader_state_dict = next(train_loader) # prefetch the very first batch of data
resume_loop_state = meta.get("loop_state", {}) if resuming else {}
progress = float(resume_loop_state.get("progress", train_loader.approx_progress if resuming else 0.0))
min_val_bpb = float(resume_loop_state.get("min_val_bpb", meta.get("val_bpb", float("inf"))))
smooth_train_loss = 0 # EMA of training loss
smooth_train_loss = float(resume_loop_state.get("smooth_train_loss", smooth_train_loss))
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
total_training_time = float(resume_loop_state.get("total_training_time", total_training_time))
step = args.resume_from_step if resuming else 0
val_bpb = float(meta.get("val_bpb", float("nan")))
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    last_step = train_loader.last_step
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())
        train_loader.last_step = last_step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the ChatCORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
        }
        task_results = {}
        for task_name in all_tasks:
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 means no limit
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100*acc:.2f}%")
        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
        chatcore_cat = centered_mean(categorical_tasks)
        print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })
        model.train()

    # save checkpoint at the end of the run, or periodically if requested
    should_save = last_step or (
        step > 0
        and step != args.resume_from_step
        and args.save_every > 0
        and step % args.save_every == 0
    )
    if should_save:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": asdict(orig_model.config),
                "wandb_run_id": getattr(wandb_run, "id", None),
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": args.total_batch_size,
                "dataloader_state_dict": train_loader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "progress": progress,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, train_loader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, train_loader.approx_progress) # only increase progress monotonically
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {train_loader.current_epoch} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": train_loader.current_epoch,
        })

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        gc.disable() # disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from etude.report import get_report
get_report().log(section="SFT", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
