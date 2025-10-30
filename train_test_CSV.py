#train_with_features.py --------------------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
import os, argparse, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import get_linear_schedule_with_warmup

from featuresdataset import StabilityWithFeaturesDataset, make_collate_fn
from bert_with_features import PLM_With_Features

import sys
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

# --------- Defaults / hparams (kept same logic) ---------
BATCH_SIZE=16
EPOCHS=10
GRAD_ACCUM_STEPS=4
WARMUP_RATIO=0.06
MAX_LEN=512
NUM_WORKERS=0
USE_TOKEN_FEATS=False
LM_NAME='facebook/esm2_t33_650M_UR50D'

def build_paths(root: str, protein: str, run_tag: str | None):
    pdir = os.path.join(root, protein)
    out_dir = os.path.join('results/dms', protein)
    os.makedirs(out_dir, exist_ok=True)
    base = 'best_model_features' if not run_tag else f'best_model_features__{run_tag}'
    return {
        'PDir':       pdir,
        'FASTA_TRAIN': os.path.join(pdir, 'train_seqs.txt'),
        'FASTA_VALID': os.path.join(pdir, 'valid_seqs.txt'),
        'X_TRAIN':     os.path.join(pdir, 'X_train_std.npy'),
        'X_VALID':     os.path.join(pdir, 'X_valid_std.npy'),
        'Y_TRAIN':     os.path.join(pdir, 'y_train_aligned.npy'),
        'Y_VALID':     os.path.join(pdir, 'y_valid_aligned.npy'),
        'WT_GEOM':     os.path.join(pdir, f'WT_geom_{protein}.npz'),
        'WT_SEQ':      os.path.join(pdir, 'wt_seq.txt'),
        'OUT_PATH':    os.path.join(out_dir, base + '.pt')
    }

def train_one(protein: str, root: str, device, lr_bert: float, lr_fusion: float,
              wd_bert: float, wd_fusion: float, run_tag: str | None):
    paths = build_paths(root, protein, run_tag)

    # --------- Data ---------
    train_dataset=StabilityWithFeaturesDataset(
        paths['FASTA_TRAIN'], paths['X_TRAIN'], paths['Y_TRAIN'],
        max_length=MAX_LEN, use_token_feats=USE_TOKEN_FEATS,
        wt_geom_path=paths['WT_GEOM'], wt_seq_path=paths['WT_SEQ'],
        esm_model_name=LM_NAME
    )
    valid_dataset=StabilityWithFeaturesDataset(
        paths['FASTA_VALID'], paths['X_VALID'], paths['Y_VALID'],
        max_length=MAX_LEN, use_token_feats=USE_TOKEN_FEATS,
        wt_geom_path=paths['WT_GEOM'], wt_seq_path=paths['WT_SEQ'],
        esm_model_name=LM_NAME
    )

    collate_fn=make_collate_fn(pad_id=train_dataset.tokenizer.pad_token_id)
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,
                            collate_fn=collate_fn,pin_memory=True,num_workers=NUM_WORKERS)
    valid_loader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,
                            collate_fn=collate_fn,pin_memory=True,num_workers=NUM_WORKERS)

    print(f"🖥️ Using device: {device}")

    feature_dim=train_dataset.features.shape[1]
    token_feat_dim=train_dataset.token_feat_dim if USE_TOKEN_FEATS else 9
    model=PLM_With_Features(feature_dim=feature_dim, bert_model_name=LM_NAME, dropout=0.1,
                           token_feat_dim=token_feat_dim, cross_layers=2, heads=4).to(device)

    # --------- Optimizer/scheduler (same logic; params now CLI-configurable) ---------
    bert_params=[]; fusion_params=[]
    for n,p in model.named_parameters():
        if n.startswith('bert.'): bert_params.append(p)
        else: fusion_params.append(p)
    opt_groups=[
        {'params': bert_params,   'lr': lr_bert,   'weight_decay': wd_bert},
        {'params': fusion_params, 'lr': lr_fusion, 'weight_decay': wd_fusion},
    ]
    optimizer=torch.optim.AdamW(opt_groups)

    total_steps=(len(train_loader)*EPOCHS)//GRAD_ACCUM_STEPS
    num_warmup=max(1, int(WARMUP_RATIO*total_steps))
    scheduler=get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)

    # --------- Train loop (unchanged) ---------
    best_spearman=-1.0
    model.train()
    print(f"\n=== Training [{protein}] | lr_bert={lr_bert} lr_fusion={lr_fusion} tag={run_tag or 'none'} ===")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_losses=[]; optimizer.zero_grad(set_to_none=True)

        for step,batch in enumerate(tqdm(train_loader,desc="Training")):
            input_ids=batch['input_ids'].to(device)
            input_mask=batch['input_mask'].to(device)
            features=batch['features'].to(device)
            targets=batch['targets'].to(device)
            per_token_feats=batch.get('per_token_feats')
            if per_token_feats is not None: per_token_feats=per_token_feats.to(device)

            loss,_=model(input_ids,input_mask,features,targets=targets,per_token_feats=per_token_feats)
            (loss/GRAD_ACCUM_STEPS).backward()

            if (step+1)%GRAD_ACCUM_STEPS==0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item())

        print(f"Train Loss: {float(np.mean(train_losses)):.4f}")

        # --------- Validate ---------
        model.eval()
        val_losses=[]; preds=[]; gold=[]
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                input_ids=batch['input_ids'].to(device)
                input_mask=batch['input_mask'].to(device)
                features=batch['features'].to(device)
                targets=batch['targets'].to(device)
                per_token_feats=batch.get('per_token_feats')
                if per_token_feats is not None: per_token_feats=per_token_feats.to(device)

                loss,pred=model(input_ids,input_mask,features,targets=targets,per_token_feats=per_token_feats)
                val_losses.append(loss.item()); preds.extend(pred.cpu().numpy().tolist()); gold.extend(targets.cpu().numpy().tolist())

        spearman = spearmanr(gold, preds).correlation or 0.0
        print(f"Val Loss: {float(np.mean(val_losses)):.4f} | 📈 Spearman: {spearman:.4f}")
        if spearman>best_spearman:
            best_spearman=spearman
            torch.save(model.state_dict(),paths['OUT_PATH'])
            print(f"💾 Saved best model → {paths['OUT_PATH']}")
        model.train()

    print(f"\n🏁 [{protein}] training complete. Best Spearman={best_spearman:.4f}")
    return paths['OUT_PATH']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='data/dms_one', help='Root with per-protein subfolders')
    ap.add_argument('--protein', required=True, help='Protein folder name (e.g., YAP1_HUMAN_Araya_2012)')
    ap.add_argument('--seed', type=int, default=42)
    # NEW: learning rates & weight decays
    ap.add_argument('--lr_bert', type=float, default=1e-4)
    ap.add_argument('--lr_fusion', type=float, default=1e-4)
    ap.add_argument('--wd_bert', type=float, default=0.01)
    ap.add_argument('--wd_fusion', type=float, default=0.01)
    # NEW: run tag to avoid checkpoint overwrite
    ap.add_argument('--run_tag', type=str, default=None)
    args=ap.parse_args()

    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pdir=os.path.join(args.root, args.protein)
    req = ['train_seqs.txt','valid_seqs.txt','X_train_std.npy','X_valid_std.npy','y_train_aligned.npy','y_valid_aligned.npy']
    missing=[f for f in req if not os.path.exists(os.path.join(pdir,f))]
    if missing:
        raise RuntimeError(f"Missing required files in {pdir}: {missing}\nRun features.py first for this protein.")

    train_one(args.protein, args.root, device,
              lr_bert=args.lr_bert, lr_fusion=args.lr_fusion,
              wd_bert=args.wd_bert, wd_fusion=args.wd_fusion,
              run_tag=args.run_tag)

if __name__=='__main__':
    main()


#TO RUN
#train_with_features.py --root data/dms_one --protein YAP1_HUMAN_Araya_2012





# ----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
#test.py
#!/usr/bin/env python3
import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from featuresdataset import StabilityWithFeaturesDataset, make_collate_fn
from bert_with_features import PLM_With_Features

def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

MAX_LEN=512
USE_TOKEN_FEATS=False
LM_NAME='facebook/esm2_t33_650M_UR50D'

def build_paths(root: str, protein: str, run_tag: str | None):
    pdir = os.path.join(root, protein)
    base = 'best_model_features' if not run_tag else f'best_model_features__{run_tag}'
    return {
        'FASTA_TEST': os.path.join(pdir, 'test_seqs.txt'),
        'X_TEST':     os.path.join(pdir, 'X_test_std.npy'),
        'Y_TEST':     os.path.join(pdir, 'y_test_aligned.npy'),
        'WT_GEOM':    os.path.join(pdir, f'WT_geom_{protein}.npz'),
        'WT_SEQ':     os.path.join(pdir, 'wt_seq.txt'),
        'MODEL_PATH': os.path.join('results/dms', protein, base + '.pt'),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='data/dms_one', help='Root with per-protein subfolders')
    ap.add_argument('--protein', required=True, help='Protein folder name (e.g., YAP1_HUMAN_Araya_2012)')
    ap.add_argument('--run_tag', type=str, default=None, help='Run tag used at training time (if any)')
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()

    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    paths = build_paths(args.root, args.protein, args.run_tag)
    for k in ['FASTA_TEST','X_TEST','Y_TEST','MODEL_PATH']:
        p = paths[k]
        if not os.path.exists(p):
            raise RuntimeError(f"Missing {k}: {p}")

    test_dataset=StabilityWithFeaturesDataset(
        paths['FASTA_TEST'], paths['X_TEST'], paths['Y_TEST'],
        max_length=MAX_LEN, use_token_feats=USE_TOKEN_FEATS,
        wt_geom_path=paths['WT_GEOM'], wt_seq_path=paths['WT_SEQ'],
        esm_model_name=LM_NAME
    )
    collate_fn=make_collate_fn(pad_id=test_dataset.tokenizer.pad_token_id)
    test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False,
                           collate_fn=collate_fn,pin_memory=True,num_workers=0)

    feature_dim=test_dataset.features.shape[1]
    token_feat_dim=test_dataset.token_feat_dim if USE_TOKEN_FEATS else 9
    model=PLM_With_Features(feature_dim=feature_dim, bert_model_name=LM_NAME,
                           dropout=0.1, token_feat_dim=token_feat_dim,
                           cross_layers=2, heads=4).to(device)
    state=torch.load(paths['MODEL_PATH'], map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds=[]; gold=[]
    with torch.no_grad():
        for batch in test_loader:
            input_ids=batch['input_ids'].to(device)
            input_mask=batch['input_mask'].to(device)
            features=batch['features'].to(device)
            targets=batch['targets'].to(device)
            per_token_feats=batch.get('per_token_feats')
            if per_token_feats is not None: per_token_feats=per_token_feats.to(device)

            p=model(input_ids,input_mask,features,per_token_feats=per_token_feats)
            preds.extend(p.cpu().numpy().tolist()); gold.extend(targets.cpu().numpy().tolist())

    spearman=spearmanr(gold,preds).correlation or 0.0
    print(f"✅ [{args.protein}] Test Spearman Correlation: {spearman:.4f} (tag={args.run_tag or 'none'})")

if __name__=='__main__':
    main()
