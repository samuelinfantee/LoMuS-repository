#test.py ----------------------------------------------------------------------------------------------------------------
import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from featuresdataset import StabilityWithFeaturesDataset, make_collate_fn
from PLM_with_features import PLM_With_Features


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


FASTA_TEST = 'data/stability/stability_test_seqs.txt'
X_TEST = 'X_test_std.npy'
Y_TEST = 'y_test_aligned.npy'
MAX_LEN = 512
LM_NAME = 'facebook/esm2_t33_650M_UR50D'
OUTPUT_DIR = 'results/stability'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_tag', type=str, default=None, help='Run tag used at training time (if any)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    base = 'best_model_features' if not args.run_tag else f'best_model_features__{args.run_tag}'
    model_path = os.path.join(OUTPUT_DIR, base + '.pt')

    for pth in [FASTA_TEST, X_TEST, Y_TEST, model_path]:
        if not os.path.exists(pth):
            raise RuntimeError(f"Missing required path: {pth}")

    test_dataset = StabilityWithFeaturesDataset(
        FASTA_TEST, X_TEST, Y_TEST,
        max_length=MAX_LEN,
        esm_model_name=LM_NAME
    )
    collate_fn = make_collate_fn(pad_id=test_dataset.tokenizer.pad_token_id)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0
    )

    feature_dim = test_dataset.features.shape[1]
    model = PLM_With_Features(
        feature_dim=feature_dim,
        PLM_model_name=LM_NAME,
        dropout=0.1
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds = []
    gold = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)

            p = model(input_ids, input_mask, features)
            preds.extend(p.cpu().numpy().tolist())
            gold.extend(targets.cpu().numpy().tolist())

    os.makedirs("figs", exist_ok=True)

    preds_arr = np.array(preds, dtype=np.float32)
    gold_arr = np.array(gold, dtype=np.float32)

    np.save("preds.npy", preds_arr)
    np.save("y_true.npy", gold_arr)

    print("Saved preds.npy and y_true.npy")
    print(f"preds shape: {preds_arr.shape}, y_true shape: {gold_arr.shape}")

    spearman = spearmanr(gold_arr, preds_arr).correlation or 0.0
    print(f"✅ Stability Test Spearman: {spearman:.4f} (tag={args.run_tag or 'none'})")


if __name__ == '__main__':
    main()
