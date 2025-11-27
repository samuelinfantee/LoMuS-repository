#features.py
#!/usr/bin/env python3
"""
Builds per-sequence features for DMS per-protein and Tsuboyama datasets while keeping the
same feature recipe/logic as the original TAPE Stability pipeline).

Each CSV must have:
  - sequence : str (AA, uppercase)
  - a numeric target column: one of {target, stability, score, y} or any first numeric column

Outputs (inside each protein folder):
  - train_seqs_v2.txt, valid_seqs_v2.txt, test_seqs_v2.txt
  - X_train_aligned_v2.npy / y_train_aligned_v2.npy
  - X_valid_aligned_v2.npy / y_valid_aligned_v2.npy
  - X_test_aligned_v2.npy  / y_test_aligned_v2.npy
  - X_train_std_v2.npy / X_valid_std_v2.npy / X_test_std_v2.npy
  - feature_names_all_v2.txt, feature_names_v2.txt
  - scaler_v2.pkl

Usage:
  1) Follow instructions from the README.
  2) Run the code (Also in instructions):
    DMS per-protein:
        python features_CSV.py

    Tsuboyama:
        python features.py --root data/dms_one --protein tsub_mega
"""

import os
import csv
import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Config (paths)
AAINDEX_PATH = './aaindex1.txt'
DMS_ROOT     = './data/dms_one'

# ---------------------------
# AAindex parser (same as original)
def parse_aaindex1(file_path):
    aaindex = {}
    aa_order = 'ARNDCQEGHILKMFPSTWYV'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    current_id = None
    values = []
    inside_entry = False
    for line in lines:
        line = line.strip()
        if line.startswith('H '):
            current_id = line[2:].strip()
            values = []
            inside_entry = True
            continue
        if inside_entry and line.startswith('I '):
            continue
        if inside_entry and line.startswith('//'):
            if current_id and len(values) == 20:
                aaindex[current_id] = dict(zip(aa_order, values))
            current_id = None
            values = []
            inside_entry = False
            continue
        if inside_entry and any(ch.isdigit() for ch in line):
            try:
                parts = [float(x) for x in line.split()]
                values.extend(parts)
            except ValueError:
                continue
    print(f"[AAindex] Parsed {len(aaindex)} features")
    return aaindex, list(aaindex.keys())

# ---------------------------
# Basis features (same recipe as original)
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    def compute_basis_features(sequence):
        analyzed = ProteinAnalysis(sequence)
        mw = analyzed.molecular_weight()
        pI = analyzed.isoelectric_point()
        gravy = analyzed.gravy()
        instability = analyzed.instability_index()
        seq_len = len(sequence)

        aa_comp = analyzed.get_amino_acids_percent()
        AI = (
            aa_comp.get('A', 0.0) * 1.0 +
            aa_comp.get('V', 0.0) * 2.9 +
            (aa_comp.get('I', 0.0) + aa_comp.get('L', 0.0)) * 3.9
        ) * 100.0

        return [seq_len, mw, pI, gravy, AI, instability]
except Exception as e:
    print(f"[WARN] Biopython not available ({e}). Using fallback basis features.")
    AA_MASS = {
        'A': 89.09,'R': 174.20,'N': 132.12,'D': 133.10,'C': 121.15,
        'Q': 146.15,'E': 147.13,'G': 75.07,'H': 155.16,'I': 131.17,
        'L': 131.17,'K': 146.19,'M': 149.21,'F': 165.19,'P': 115.13,
        'S': 105.09,'T': 119.12,'W': 204.23,'Y': 181.19,'V': 117.15
    }
    def compute_basis_features(sequence):
        seq_len = len(sequence)
        mw = sum(AA_MASS.get(a, 110.0) for a in sequence)
        pI = 7.0
        gravy = 0.0
        AI = 0.0
        instability = 40.0
        return [seq_len, mw, pI, gravy, AI, instability]

basis_names = ['seq_len', 'molecular_weight', 'isoelectric_point', 'gravy', 'aliphatic_index', 'instability_index']

# ---------------------------
# Vectorize one sequence (AAindex mean + basis) — same logic as original
def sequence_to_vector(sequence, aaindex, aaindex_keys):
    aa_features = []
    for aa in sequence:
        vals = [aaindex[k].get(aa, 0.0) for k in aaindex_keys]
        aa_features.append(vals)
    mean_aaindex = np.mean(aa_features, axis=0) if aa_features else np.zeros(len(aaindex_keys), dtype=np.float32)
    basis = compute_basis_features(sequence)
    return np.concatenate([mean_aaindex, np.array(basis, dtype=np.float32)])

# ---------------------------
def _read_csv(path):
    """Read a CSV/TSV with at least 'sequence' and one numeric target column."""
    with open(path, 'r', newline='') as f:
        sample = f.read(2048); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        rows = [r for r in reader]

    seqs, ys = [], []
    if not rows:
        return seqs, np.asarray(ys, dtype=np.float32)

    target_keys = ['target', 'stability', 'score', 'y']
    for r in rows:
        s = (r.get('sequence') or r.get('seq') or r.get('primary') or '').strip().upper()
        if not s:
            continue
        yval = None
        for k in target_keys:
            if r.get(k) not in (None, ''):
                try:
                    yval = float(r[k]); break
                except ValueError:
                    pass
        if yval is None:
            # fallback: first numeric-looking column that's not an obvious ID field
            for c, v in r.items():
                if c is None:
                    continue
                cl = c.lower()
                if cl in {'protein','name','id','uniprot','pdb','variant','mutation','mut','pos','position','wt','wt_aa','mt','mt_aa'}:
                    continue
                try:
                    yval = float(v); break
                except:
                    continue
        if yval is None:
            continue
        seqs.append(s); ys.append(yval)

    return seqs, np.asarray(ys, dtype=np.float32)

def _write_fasta_like(path, sequences):
    with open(path, 'w') as f:
        for s in sequences:
            f.write(s + '\n')

def _process_protein(protein_dir, aaindex, aaindex_keys):
    protein = os.path.basename(protein_dir.rstrip('/'))
    print(f"\n=== [{protein}] ===")

    # required CSVs
    req = ['train.csv','valid.csv','test.csv']
    for fn in req:
        if not os.path.exists(os.path.join(protein_dir, fn)):
            raise FileNotFoundError(f"Missing {fn} under {protein_dir}")

    # load splits
    tr_seqs, y_tr = _read_csv(os.path.join(protein_dir, 'train.csv'))
    va_seqs, y_va = _read_csv(os.path.join(protein_dir, 'valid.csv'))
    te_seqs, y_te = _read_csv(os.path.join(protein_dir, 'test.csv'))

    # feature names (ALL cols kept, same policy as original)
    feature_names_all = aaindex_keys + basis_names
    with open(os.path.join(protein_dir, 'feature_names_all.txt'), 'w') as f:
        for n in feature_names_all: f.write(n + '\n')
    with open(os.path.join(protein_dir, 'feature_names.txt'), 'w') as f:
        for n in feature_names_all: f.write(n + '\n')
    print(f"[OUT] feature_names_* written ({len(feature_names_all)} names)")

    # vectorize with identical recipe
    def vec_many(seqs):
        out = []
        for s in tqdm(seqs, desc=" featurize"):
            out.append(sequence_to_vector(s, aaindex, aaindex_keys))
        return np.asarray(out, dtype=np.float32)

    X_tr_raw = vec_many(tr_seqs); X_va_raw = vec_many(va_seqs); X_te_raw = vec_many(te_seqs)

    # save fasta-like lists for dataset class
    _write_fasta_like(os.path.join(protein_dir, 'train_seqs.txt'), tr_seqs)
    _write_fasta_like(os.path.join(protein_dir, 'valid_seqs.txt'), va_seqs)
    _write_fasta_like(os.path.join(protein_dir, 'test_seqs.txt'),  te_seqs)

    # save aligned raw
    np.save(os.path.join(protein_dir, 'X_train_aligned.npy'), X_tr_raw); np.save(os.path.join(protein_dir, 'y_train_aligned.npy'), y_tr)
    np.save(os.path.join(protein_dir, 'X_valid_aligned.npy'), X_va_raw); np.save(os.path.join(protein_dir, 'y_valid_aligned.npy'), y_va)
    np.save(os.path.join(protein_dir, 'X_test_aligned.npy'),  X_te_raw); np.save(os.path.join(protein_dir, 'y_test_aligned.npy'),  y_te)
    print("[OUT] Saved aligned arrays.")

    # standardize (same as original)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)
    X_te = scaler.transform(X_te_raw)
    np.save(os.path.join(protein_dir, 'X_train_std.npy'), X_tr)
    np.save(os.path.join(protein_dir, 'X_valid_std.npy'), X_va)
    np.save(os.path.join(protein_dir, 'X_test_std.npy'),  X_te)
    with open(os.path.join(protein_dir, 'scaler.pkl'), 'wb') as f:
        pkl.dump(scaler, f)
    print("[OUT] Saved standardized arrays + scaler.pkl")

    # sanity
    for name, y in [('train', y_tr), ('valid', y_va), ('test', y_te)]:
        if len(y) == 0:
            print(f"  {name}: n=0  (WARNING: empty split)")
        else:
            print(f"  {name}: n={len(y):5d}  mean={np.mean(y):.6f}  std={np.std(y):.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DMS_ROOT, help='Root folder with per-protein subfolders')
    ap.add_argument('--protein', nargs='*', help='Specific protein folder names to process')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"Root not found: {args.root}")

    aaindex, aaindex_keys = parse_aaindex1(AAINDEX_PATH)

    proteins = []
    if args.protein:
        proteins = [p for p in args.protein if os.path.isdir(os.path.join(args.root, p))]
        if not proteins:
            raise RuntimeError("No valid protein folders found from --protein")
    else:
        proteins = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
        if not proteins:
            raise RuntimeError(f"No protein subfolders under {args.root}")

    for p in proteins:
        _process_protein(os.path.join(args.root, p), aaindex, aaindex_keys)

    print("\n✅ Done. Training should point to per-protein X_*_std.npy inside each folder.")

if __name__ == '__main__':
    main()
