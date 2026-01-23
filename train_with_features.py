#train_with_features.py --------------------------------------------------------------------------------------------------------------
import os, argparse, random, numpy as np, torch, torch.nn as nn
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


# --------- Defaults ---------
FASTA_TRAIN = 'data/stability/stability_train_seqs.txt'
FASTA_VALID = 'data/stability/stability_valid_seqs.txt'
X_TRAIN = 'X_train_std.npy'
X_VALID = 'X_valid_std.npy'
Y_TRAIN = 'y_train_aligned.npy'
Y_VALID = 'y_valid_aligned.npy'

BATCH_SIZE = 8
DEFAULT_EPOCHS = 10
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO = 0.06

LM_NAME = 'facebook/esm2_t33_650M_UR50D'
OUTPUT_DIR = 'results/stability'
MAX_LEN = 512
NUM_WORKERS = 0


def train_one(device,
              lr_PLM: float,
              lr_fusion: float,
              wd_PLM: float,
              wd_fusion: float,
              run_tag: str | None,
              epochs: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = 'best_model_features' if not run_tag else f'best_model_features__{run_tag}'
    out_path = os.path.join(OUTPUT_DIR, base + '.pt')

    # --------- Data ---------
    train_dataset = StabilityWithFeaturesDataset(
        FASTA_TRAIN, X_TRAIN, Y_TRAIN,
        max_length=MAX_LEN,
        esm_model_name=LM_NAME
    )
    valid_dataset = StabilityWithFeaturesDataset(
        FASTA_VALID, X_VALID, Y_VALID,
        max_length=MAX_LEN,
        esm_model_name=LM_NAME
    )

    collate_fn = make_collate_fn(pad_id=train_dataset.tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    print(f"🖥️ Using device: {device}")

    feature_dim = train_dataset.features.shape[1]
    model = PLM_With_Features(
        feature_dim=feature_dim,
        PLM_model_name=LM_NAME,
        dropout=0.1,
        use_lora=True,
        ).to(device)

    # --------- Two LR optimizer (PLM vs head) ---------
    PLM_params = []
    fusion_params = []
    for n, p in model.named_parameters():
        if n.startswith('PLM.'):
            PLM_params.append(p)
        else:
            fusion_params.append(p)
    opt_groups = [
        {'params': PLM_params, 'lr': lr_PLM, 'weight_decay': wd_PLM},
        {'params': fusion_params, 'lr': lr_fusion, 'weight_decay': wd_fusion},
    ]
    optimizer = torch.optim.AdamW(opt_groups)

    total_steps = (len(train_loader) * epochs) // GRAD_ACCUM_STEPS
    num_warmup = max(1, int(WARMUP_RATIO * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)

    # --------- Train ---------
    best_spearman = -1.0
    model.train()
    print(
        f"\n=== Training [stability] | lr_PLM={lr_PLM} lr_fusion={lr_fusion} "
        f"wd_PLM={wd_PLM} wd_fusion={wd_fusion} tag={run_tag or 'none'} ==="
    )
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_losses = []
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)

            loss, _ = model(input_ids, input_mask, features, targets=targets)
            (loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item())

        print(f"Train Loss: {float(np.mean(train_losses)):.4f}")

        # --------- Validate ---------
        model.eval()
        val_losses = []
        preds = []
        gold = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                input_mask = batch['input_mask'].to(device)
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)

                loss, pred = model(input_ids, input_mask, features, targets=targets)
                val_losses.append(loss.item())
                preds.extend(pred.cpu().numpy().tolist())
                gold.extend(targets.cpu().numpy().tolist())

        spearman = spearmanr(gold, preds).correlation or 0.0
        print(f"Val Loss: {float(np.mean(val_losses)):.4f} | 📈 Spearman: {spearman:.4f}")
        if spearman > best_spearman:
            best_spearman = spearman
            torch.save(model.state_dict(), out_path)
            print(f"💾 Saved best model → {out_path}")
        model.train()

    print(f"\n🏁 Training complete. Best Spearman={best_spearman:.4f}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    # learning rates and weight decays (for sweep)
    ap.add_argument('--lr_PLM', type=float, default=1e-4)
    ap.add_argument('--lr_fusion', type=float, default=1e-4)
    ap.add_argument('--wd_PLM', type=float, default=0.01)
    ap.add_argument('--wd_fusion', type=float, default=0.01)
    # run tag to make checkpoint names unique
    ap.add_argument('--run_tag', type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # basic file sanity check
    req = [FASTA_TRAIN, FASTA_VALID, X_TRAIN, X_VALID, Y_TRAIN, Y_VALID]
    miss = [p for p in req if not os.path.exists(p)]
    if miss:
        raise RuntimeError(f"Missing required files: {miss}\nRun features_FASTA.py first for stability.")

    train_one(
        device,
        lr_PLM=args.lr_PLM,
        lr_fusion=args.lr_fusion,
        wd_PLM=args.wd_PLM,
        wd_fusion=args.wd_fusion,
        run_tag=args.run_tag,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()
