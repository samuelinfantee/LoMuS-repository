#integrated_gradients.py ------------------------------------------------
"""
Integrated Gradients analysis for the physicochemical feature branch of LoMuS.

What this script does:
- Loads a trained PLM_With_Features checkpoint.
- Loads one split (train/valid/test) for a given protein dataset.
- Computes Integrated Gradients with respect to the raw standardized physicochemical input vector
  while keeping the pooled PLM sequence representation fixed for each sample.
- Saves per-sample attributions and aggregated feature rankings.

Example:
python integrated_gradients.py \
    --root data/dms_one \
    --protein PIN1_HUMAN_Tsuboyama_2023_1I6C \
    --split test \
    --checkpoint results/stability/checkpoints/best__pin1_run.pt \
    --run_tag pin1_ig \
    --output_dir results/stability/integrated_gradients
"""

import os
import json
import argparse
import random
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from featuresdataset import StabilityWithFeaturesDataset, make_collate_fn
from PLM_with_features import PLM_With_Features


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_use_lora_from_state_dict(state_dict):
    return any("lora_" in k for k in state_dict.keys())


def load_feature_names(data_dir, expected_dim):
    """
    Try to load feature names saved by features_CSV.py.
    Falls back to generic names if not found.
    """
    candidates = [
        os.path.join(data_dir, "feature_names.txt"),
        os.path.join(data_dir, "feature_names_all.txt"),
        os.path.join(data_dir, "feature_names_v2.txt"),
        os.path.join(data_dir, "feature_names_all_v2.txt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                names = [ln.strip() for ln in f if ln.strip()]
            if len(names) == expected_dim:
                return names

    return [f"feature_{i}" for i in range(expected_dim)]


def get_split_paths(root, protein, split):
    data_dir = os.path.join(root, protein)

    if split == "train":
        fasta_path = os.path.join(data_dir, "train_seqs.txt")
        x_path = os.path.join(data_dir, "X_train_std.npy")
        y_path = os.path.join(data_dir, "y_train_aligned.npy")
    elif split == "valid":
        fasta_path = os.path.join(data_dir, "valid_seqs.txt")
        x_path = os.path.join(data_dir, "X_valid_std.npy")
        y_path = os.path.join(data_dir, "y_valid_aligned.npy")
    elif split == "test":
        fasta_path = os.path.join(data_dir, "test_seqs.txt")
        x_path = os.path.join(data_dir, "X_test_std.npy")
        y_path = os.path.join(data_dir, "y_test_aligned.npy")
    else:
        raise ValueError(f"Unsupported split: {split}")

    return data_dir, fasta_path, x_path, y_path


def compute_integrated_gradients_for_batch(model, input_ids, input_mask, features, steps=64, baseline=None):
    """
    Compute Integrated Gradients for the raw standardized physicochemical features.

    Important:
    - The pooled PLM sequence representation is kept fixed for the sample.
    - IG is computed only through the physicochemical branch:
          features -> feature_proj -> regressor
    - This isolates the contribution of physicochemical inputs conditioned on the sequence embedding.

    Args:
        model: PLM_With_Features in eval mode
        input_ids: (B, L)
        input_mask: (B, L)
        features: (B, F)
        steps: number of IG integration steps
        baseline: tensor of shape (B, F) or None. If None, uses zero baseline.

    Returns:
        attributions: (B, F)
    """
    # Encode once so the PLM branch is fixed for attribution
    with torch.no_grad():
        pooled = model.encode_sequence(input_ids, input_mask)

    if baseline is None:
        baseline = torch.zeros_like(features)

    # Difference from baseline to actual input
    delta = features - baseline

    total_gradients = torch.zeros_like(features)

    # Use Riemann summation over interpolation path
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=features.device)[1:]

    for alpha in alphas:
        interpolated = baseline + alpha * delta
        interpolated.requires_grad_(True)

        preds = model.predict_from_pooled_and_features(pooled, interpolated)
        grads = torch.autograd.grad(
            outputs=preds.sum(),
            inputs=interpolated,
            create_graph=False,
            retain_graph=False
        )[0]

        total_gradients += grads

    avg_gradients = total_gradients / float(steps)
    attributions = delta * avg_gradients
    return attributions


def save_feature_importance_csv(path, feature_names, signed_mean, abs_mean, abs_std, abs_median):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature_name", "mean_signed_ig", "mean_abs_ig", "std_abs_ig", "median_abs_ig"])
        order = np.argsort(-abs_mean)
        for rank, idx in enumerate(order, start=1):
            writer.writerow([
                rank,
                feature_names[idx],
                float(signed_mean[idx]),
                float(abs_mean[idx]),
                float(abs_std[idx]),
                float(abs_median[idx]),
            ])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--protein", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])

    ap.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    ap.add_argument("--run_tag", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)

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
    ap.add_argument("--lora_targets", type=str, default="auto")

    ap.add_argument("--output_dir", type=str, default="results/stability/integrated_gradients")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    data_dir, fasta_path, x_path, y_path = get_split_paths(args.root, args.protein, args.split)
    required = [fasta_path, x_path, y_path, args.checkpoint]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Missing required paths: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = StabilityWithFeaturesDataset(
        fasta_path=fasta_path,
        features_path=x_path,
        labels_path=y_path,
        max_length=args.max_len,
        esm_model_name=args.lm_name
    )
    collate_fn = make_collate_fn(pad_id=dataset.tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=args.num_workers
    )

    feature_dim = dataset.features.shape[1]
    feature_names = load_feature_names(data_dir, feature_dim)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    inferred_use_lora = infer_use_lora_from_state_dict(checkpoint) if args.use_lora is None else args.use_lora

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

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    all_attr = []
    all_targets = []

    for batch in tqdm(loader, desc=f"IG {args.split}"):
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)

        attributions = compute_integrated_gradients_for_batch(
            model=model,
            input_ids=input_ids,
            input_mask=input_mask,
            features=features,
            steps=args.steps,
            baseline=None  # zero baseline in standardized space
        )

        all_attr.append(attributions.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    all_attr = np.concatenate(all_attr, axis=0).astype(np.float32)
    all_abs_attr = np.abs(all_attr).astype(np.float32)
    all_targets = np.concatenate(all_targets, axis=0).astype(np.float32)

    signed_mean = all_attr.mean(axis=0)
    abs_mean = all_abs_attr.mean(axis=0)
    abs_std = all_abs_attr.std(axis=0)
    abs_median = np.median(all_abs_attr, axis=0)

    # Save arrays
    np.save(os.path.join(args.output_dir, f"{args.run_tag}__ig_attributions.npy"), all_attr)
    np.save(os.path.join(args.output_dir, f"{args.run_tag}__ig_abs_attributions.npy"), all_abs_attr)
    np.save(os.path.join(args.output_dir, f"{args.run_tag}__targets.npy"), all_targets)

    # Save ranking CSV
    csv_path = os.path.join(args.output_dir, f"{args.run_tag}__ig_feature_importance.csv")
    save_feature_importance_csv(
        csv_path,
        feature_names=feature_names,
        signed_mean=signed_mean,
        abs_mean=abs_mean,
        abs_std=abs_std,
        abs_median=abs_median
    )

    # Save top-20 text summary
    order = np.argsort(-abs_mean)
    top20_path = os.path.join(args.output_dir, f"{args.run_tag}__ig_top20.txt")
    with open(top20_path, "w") as f:
        f.write("Top 20 features by mean absolute Integrated Gradients attribution\n")
        f.write("===============================================================\n")
        for rank, idx in enumerate(order[:20], start=1):
            f.write(
                f"{rank:02d}. {feature_names[idx]} | "
                f"mean_abs_ig={abs_mean[idx]:.8f} | "
                f"mean_signed_ig={signed_mean[idx]:.8f} | "
                f"std_abs_ig={abs_std[idx]:.8f} | "
                f"median_abs_ig={abs_median[idx]:.8f}\n"
            )

    # Basic branch-level summary
    basis_feature_names = {
        "seq_len",
        "molecular_weight",
        "isoelectric_point",
        "gravy",
        "aliphatic_index",
        "instability_index",
    }
    basis_mask = np.array([name in basis_feature_names for name in feature_names], dtype=bool)
    aaindex_mask = ~basis_mask

    summary = {
        "run_tag": args.run_tag,
        "protein": args.protein,
        "split": args.split,
        "num_samples": int(all_attr.shape[0]),
        "feature_dim": int(all_attr.shape[1]),
        "steps": int(args.steps),
        "checkpoint": args.checkpoint,
        "use_lora": bool(inferred_use_lora),
        "basis_feature_abs_mean_sum": float(abs_mean[basis_mask].sum()) if basis_mask.any() else 0.0,
        "aaindex_feature_abs_mean_sum": float(abs_mean[aaindex_mask].sum()) if aaindex_mask.any() else 0.0,
        "top_20_features": [
            {
                "rank": int(rank),
                "feature_name": feature_names[idx],
                "mean_abs_ig": float(abs_mean[idx]),
                "mean_signed_ig": float(signed_mean[idx]),
            }
            for rank, idx in enumerate(order[:20], start=1)
        ],
        "files": {
            "ig_attributions": os.path.join(args.output_dir, f"{args.run_tag}__ig_attributions.npy"),
            "ig_abs_attributions": os.path.join(args.output_dir, f"{args.run_tag}__ig_abs_attributions.npy"),
            "targets": os.path.join(args.output_dir, f"{args.run_tag}__targets.npy"),
            "feature_importance_csv": csv_path,
            "top20_txt": top20_path,
        }
    }

    summary_path = os.path.join(args.output_dir, f"{args.run_tag}__ig_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nIntegrated Gradients analysis completed.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {top20_path}")
    print(f"Saved: {summary_path}")

    print("\nTop 10 features by mean absolute attribution:")
    for rank, idx in enumerate(order[:10], start=1):
        print(
            f"{rank:02d}. {feature_names[idx]} | "
            f"mean_abs_ig={abs_mean[idx]:.8f} | "
            f"mean_signed_ig={signed_mean[idx]:.8f}"
        )


if __name__ == "__main__":
    main()
