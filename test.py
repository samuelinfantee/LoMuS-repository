#test.py ----------------------------------------------------------------------------------------------------------------
import os, argparse, json, time, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from featuresdataset import StabilityWithFeaturesDataset, make_collate_fn
from PLM_with_features import PLM_With_Features


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def cuda_mem_stats_gb():
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserv = torch.cuda.memory_reserved() / (1024**3)
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserv = torch.cuda.max_memory_reserved() / (1024**3)
    return {
        "alloc_gb": alloc,
        "reserved_gb": reserv,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserv
    }


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def infer_use_lora_from_state_dict(state_dict):
    return any("lora_" in k for k in state_dict.keys())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", type=str, required=True)       # e.g. data/dms_one
    ap.add_argument("--protein", type=str, required=True)    # e.g. tsub_mega
    ap.add_argument("--run_tag", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--lm_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--dropout", type=float, default=0.1)

    # LoRA controls
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--no_lora", dest="use_lora", action="store_false")
    ap.set_defaults(use_lora=None)

    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    ap.set_defaults(freeze_backbone=True)

    ap.add_argument("--lora_r", type=int, default=9)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="auto")  # or "q_proj,k_proj,v_proj,out_proj"

    ap.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Directory containing best__<run_tag>.pt. If omitted, inferred from --log_dir parent/checkpoints.")
    ap.add_argument("--log_dir", type=str, required=True)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.log_dir, exist_ok=True)

    # Resolve dataset paths from root/protein
    data_dir = os.path.join(args.root, args.protein)
    fasta_test = os.path.join(data_dir, "test_seqs.txt")
    x_test = os.path.join(data_dir, "X_test_std.npy")
    y_test = os.path.join(data_dir, "y_test_aligned.npy")

    # Resolve checkpoint location
    if args.checkpoint_dir is not None:
        ckpt_dir = args.checkpoint_dir
    else:
        # Matches your SLURM layout:
        # RUN_DIR/logs -> RUN_DIR/checkpoints
        run_dir = os.path.dirname(args.log_dir.rstrip("/"))
        ckpt_dir = os.path.join(run_dir, "checkpoints")

    model_path = os.path.join(ckpt_dir, f"best__{args.run_tag}.pt")

    req = [fasta_test, x_test, y_test, model_path]
    miss = [p for p in req if not os.path.exists(p)]
    if miss:
        raise RuntimeError(f"Missing required paths: {miss}")

    test_dataset = StabilityWithFeaturesDataset(
        fasta_test,
        x_test,
        y_test,
        max_length=args.max_len,
        esm_model_name=args.lm_name
    )
    collate_fn = make_collate_fn(pad_id=test_dataset.tokenizer.pad_token_id)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=args.num_workers
    )

    feature_dim = test_dataset.features.shape[1]

    # Load checkpoint first to infer whether LoRA was used, unless explicitly provided
    state = torch.load(model_path, map_location="cpu")
    inferred_use_lora = infer_use_lora_from_state_dict(state) if args.use_lora is None else args.use_lora

    # parse targets like "q_proj,k_proj,v_proj,out_proj"
    lora_targets = None
    if args.lora_targets and args.lora_targets.lower() != "auto":
        lora_targets = tuple([s.strip() for s in args.lora_targets.split(",") if s.strip()])

    model = PLM_With_Features(
        feature_dim=feature_dim,
        PLM_model_name=args.lm_name,
        dropout=args.dropout,
        use_lora=inferred_use_lora,
        freeze_backbone=args.freeze_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    preds = []
    gold = []
    losses = []

    n_seq = 0
    n_tok = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            maybe_sync(device)
            loss, pred = model(input_ids, input_mask, features, targets=targets)
            maybe_sync(device)

            losses.append(loss.item())
            preds.extend(pred.detach().cpu().numpy().tolist())
            gold.extend(targets.detach().cpu().numpy().tolist())

            n_seq += input_ids.size(0)
            n_tok += int(input_mask.sum().item())

    t1 = time.perf_counter()
    test_dt = max(1e-9, t1 - t0)

    preds_arr = np.array(preds, dtype=np.float32)
    gold_arr = np.array(gold, dtype=np.float32)

    pred_path = os.path.join(args.log_dir, f"preds__{args.run_tag}.npy")
    gold_path = os.path.join(args.log_dir, f"y_true__{args.run_tag}.npy")
    np.save(pred_path, preds_arr)
    np.save(gold_path, gold_arr)

    test_loss = float(np.mean(losses)) if losses else float("nan")
    test_spearman = spearmanr(gold_arr, preds_arr).correlation or 0.0
    seq_per_s = n_seq / test_dt
    tok_per_s = n_tok / test_dt
    mem = cuda_mem_stats_gb()

    print(f"Saved predictions to: {pred_path}")
    print(f"Saved labels to:      {gold_path}")
    print(f"preds shape: {preds_arr.shape}, y_true shape: {gold_arr.shape}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test Spearman: {test_spearman:.4f}")
    print(f"Inference throughput: {seq_per_s:.2f} seq/s | {tok_per_s:.2f} tok/s")
    if mem:
        print(f"GPU peak alloc={mem['peak_alloc_gb']:.2f} GB peak reserved={mem['peak_reserved_gb']:.2f} GB")

    test_metrics = {
        "run_tag": args.run_tag,
        "protein": args.protein,
        "root": args.root,
        "checkpoint": model_path,
        "lm_name": args.lm_name,
        "use_lora": bool(inferred_use_lora),
        "freeze_backbone": args.freeze_backbone,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets_arg": args.lora_targets,
        "test_loss": test_loss,
        "test_spearman": float(test_spearman),
        "test_seconds": test_dt,
        "infer_seq_per_s": seq_per_s,
        "infer_tok_per_s": tok_per_s,
        "gpu_mem_gb": mem,
        "pred_path": pred_path,
        "y_true_path": gold_path,
        "num_examples": int(len(gold_arr)),
    }

    test_json = os.path.join(args.log_dir, f"test__{args.run_tag}.json")
    with open(test_json, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved test metrics JSON to: {test_json}")


if __name__ == '__main__':
    main()
