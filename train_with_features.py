import os, argparse, random, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import get_linear_schedule_with_warmup

from featuresdataset import StabilityWithFeaturesDataset, make_collate_fn
from PLM_with_features import PLM_With_Features


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def sizeof_optimizer_state_bytes(optimizer: torch.optim.Optimizer):
    # Approx size of all tensors kept in optimizer state (AdamW moments, etc.)
    nbytes = 0
    for st in optimizer.state.values():
        if isinstance(st, dict):
            for v in st.values():
                if torch.is_tensor(v):
                    nbytes += v.numel() * v.element_size()
    return nbytes


def cuda_mem_stats_gb():
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserv = torch.cuda.memory_reserved() / (1024**3)
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserv = torch.cuda.max_memory_reserved() / (1024**3)
    return {"alloc_gb": alloc, "reserved_gb": reserv, "peak_alloc_gb": peak_alloc, "peak_reserved_gb": peak_reserv}


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_inference(model, loader, device, warmup_batches=5, timed_batches=30):
    model.eval()
    it = iter(loader)

    # warmup
    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            feats = batch["features"].to(device)
            _ = model(input_ids, input_mask, feats, targets=None)

    # timed
    n_seq = 0
    n_tok = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(timed_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            feats = batch["features"].to(device)

            maybe_sync(device)
            _ = model(input_ids, input_mask, feats, targets=None)
            maybe_sync(device)

            n_seq += input_ids.size(0)
            n_tok += int(input_mask.sum().item())

    t1 = time.perf_counter()
    dt = max(1e-9, t1 - t0)
    return {"infer_seconds": dt, "infer_seq_per_s": n_seq / dt, "infer_tok_per_s": n_tok / dt}


def train_one(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Resolve dataset paths from root/protein (matches your SLURM layout)
    # Expected files in: {root}/{protein}/train_seqs.txt etc
    data_dir = os.path.join(args.root, args.protein)
    fasta_train = os.path.join(data_dir, "train_seqs.txt")
    fasta_valid = os.path.join(data_dir, "valid_seqs.txt")
    x_train = os.path.join(data_dir, "X_train_std.npy")
    x_valid = os.path.join(data_dir, "X_valid_std.npy")
    y_train = os.path.join(data_dir, "y_train_aligned.npy")
    y_valid = os.path.join(data_dir, "y_valid_aligned.npy")

    req = [fasta_train, fasta_valid, x_train, x_valid, y_train, y_valid]
    miss = [p for p in req if not os.path.exists(p)]
    if miss:
        raise RuntimeError(f"Missing required files: {miss}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = StabilityWithFeaturesDataset(fasta_train, x_train, y_train, max_length=args.max_len, esm_model_name=args.lm_name)
    valid_dataset = StabilityWithFeaturesDataset(fasta_valid, x_valid, y_valid, max_length=args.max_len, esm_model_name=args.lm_name)

    collate_fn = make_collate_fn(pad_id=train_dataset.tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              pin_memory=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                              pin_memory=True, num_workers=args.num_workers)

    feature_dim = train_dataset.features.shape[1]

    # parse targets like "q_proj,k_proj,v_proj,out_proj"
    lora_targets = None
    if args.lora_targets and args.lora_targets.lower() != "auto":
        lora_targets = tuple([s.strip() for s in args.lora_targets.split(",") if s.strip()])

    model = PLM_With_Features(
        feature_dim=feature_dim,
        PLM_model_name=args.lm_name,
        dropout=args.dropout,
        use_lora=args.use_lora,
        freeze_backbone=args.freeze_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
    ).to(device)

    total_p, trainable_p = count_params(model)
    print(f"Params: total={total_p:,}  trainable={trainable_p:,}  ({100.0*trainable_p/total_p:.4f}%)")

    # Optimizer groups: ONLY trainable params (important for fair optimizer state + speed)
    plm_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("PLM."):
            plm_params.append(p)
        else:
            head_params.append(p)

    opt_groups = [
        {"params": plm_params, "lr": args.lr_plm, "weight_decay": args.wd_plm},
        {"params": head_params, "lr": args.lr_head, "weight_decay": args.wd_head},
    ]
    optimizer = torch.optim.AdamW(opt_groups)

    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    num_warmup = max(1, int(args.warmup_ratio * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)

    # Log file (one per run)
    tag = args.run_tag
    log_path = os.path.join(args.log_dir, f"{tag}.jsonl")
    ckpt_path = os.path.join(args.output_dir, f"best__{tag}.pt")

    # Write run header
    run_cfg = {
        "run_tag": tag,
        "protein": args.protein,
        "root": args.root,
        "lm_name": args.lm_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr_plm": args.lr_plm,
        "lr_head": args.lr_head,
        "wd_plm": args.wd_plm,
        "wd_head": args.wd_head,
        "use_lora": args.use_lora,
        "freeze_backbone": args.freeze_backbone,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets_arg": args.lora_targets,
        "lora_targets_used": getattr(model, "lora_targets_used", None),
        "trainable_params": trainable_p,
        "total_params": total_p,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps({"event": "run_start", **run_cfg}) + "\n")

    best_spearman = -1.0
    model.train()

    for epoch in range(args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---------------- Train ----------------
        epoch_t0 = time.perf_counter()
        train_losses = []
        n_seq = 0
        n_tok = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train e{epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            feats = batch["features"].to(device)
            targets = batch["targets"].to(device)

            maybe_sync(device)
            loss, _ = model(input_ids, input_mask, feats, targets=targets)
            (loss / args.grad_accum).backward()
            maybe_sync(device)

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item())
            n_seq += input_ids.size(0)
            n_tok += int(input_mask.sum().item())

        epoch_t1 = time.perf_counter()
        train_dt = max(1e-9, epoch_t1 - epoch_t0)

        # ---------------- Validate ----------------
        val_t0 = time.perf_counter()
        model.eval()
        val_losses = []
        preds, gold = [], []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Val e{epoch+1}"):
                input_ids = batch["input_ids"].to(device)
                input_mask = batch["input_mask"].to(device)
                feats = batch["features"].to(device)
                targets = batch["targets"].to(device)

                loss, pred = model(input_ids, input_mask, feats, targets=targets)
                val_losses.append(loss.item())
                preds.extend(pred.detach().cpu().numpy().tolist())
                gold.extend(targets.detach().cpu().numpy().tolist())

        val_t1 = time.perf_counter()
        val_dt = max(1e-9, val_t1 - val_t0)

        spearman = spearmanr(gold, preds).correlation or 0.0
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        mem = cuda_mem_stats_gb()
        seq_per_s = n_seq / train_dt
        tok_per_s = n_tok / train_dt

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} spearman={spearman:.4f}")
        if mem:
            print(f"GPU peak alloc={mem['peak_alloc_gb']:.2f} GB peak reserved={mem['peak_reserved_gb']:.2f} GB | train_throughput={seq_per_s:.1f} seq/s ({tok_per_s:.0f} tok/s)")

        # Save best
        if spearman > best_spearman:
            best_spearman = spearman
            torch.save(model.state_dict(), ckpt_path)

        # Optimizer state size after at least one step
        opt_state_bytes = sizeof_optimizer_state_bytes(optimizer)

        # Log epoch event
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "event": "epoch_end",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "spearman": float(spearman),
                "train_seconds": train_dt,
                "val_seconds": val_dt,
                "train_seq_per_s": seq_per_s,
                "train_tok_per_s": tok_per_s,
                "gpu_mem_gb": mem,
                "optimizer_state_gb": opt_state_bytes / (1024**3),
                "best_spearman_so_far": float(best_spearman),
            }) + "\n")

        model.train()

    # Optional inference benchmark (valid set)
    infer_bench = None
    if args.bench_infer:
        infer_bench = bench_inference(model, valid_loader, device)

    with open(log_path, "a") as f:
        f.write(json.dumps({
            "event": "run_end",
            "best_spearman": float(best_spearman),
            "checkpoint": ckpt_path,
            "infer_bench": infer_bench
        }) + "\n")

    print(f"Done. best_spearman={best_spearman:.4f} ckpt={ckpt_path}")
    return ckpt_path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", type=str, required=True)       # e.g. data/dms_one
    ap.add_argument("--protein", type=str, required=True)    # e.g. PIN1_HUMAN_Tsuboyama_2023_1I6C

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--lm_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr_plm", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--wd_plm", type=float, default=0.01)
    ap.add_argument("--wd_head", type=float, default=0.01)

    # LoRA controls
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--no_lora", dest="use_lora", action="store_false")
    ap.set_defaults(use_lora=True)

    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    ap.set_defaults(freeze_backbone=True)

    ap.add_argument("--lora_r", type=int, default=9)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="auto")  # or "q_proj,k_proj,v_proj,out_proj"

    ap.add_argument("--output_dir", type=str, default="results/stability/checkpoints")
    ap.add_argument("--log_dir", type=str, default="results/stability/logs")
    ap.add_argument("--run_tag", type=str, required=True)

    ap.add_argument("--bench_infer", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)
    train_one(args)


if __name__ == "__main__":
    main()
