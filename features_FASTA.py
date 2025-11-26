#features_FASTA.py ------------------------------------------------------------------------------------------------------------------
"""
Builds per-sequence features for the TAPE Stability task in EXACT FASTA order.
Adds AAindex means + simple basis features.

Outputs (current directory):
- X_train_aligned.npy, y_train_aligned.npy
- X_valid_aligned.npy, y_valid_aligned.npy
- X_test_aligned.npy,  y_test_aligned.npy
- X_train_std.npy, X_valid_std.npy, X_test_std.npy
- feature_names_all.txt, feature_names.txt (same; we keep all cols)
- scaler.pkl

Usage:
python features_FASTA.py
"""

import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle as pkl

# ---------------------------
# Config (paths for Stability)
AAINDEX_PATH = './aaindex1.txt'

TRAIN_FASTA  = './data/stability/stability_train_seqs.txt'
VALID_FASTA  = './data/stability/stability_valid_seqs.txt'
TEST_FASTA   = './data/stability/stability_test_seqs.txt'

TRAIN_LMDB   = './data/stability/stability_train.lmdb'
VALID_LMDB   = './data/stability/stability_valid.lmdb'
TEST_LMDB    = './data/stability/stability_test.lmdb'

# ---------------------------
# AAindex parser
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
# Basis features (same recipe as before)
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
# Vectorize one sequence (AAindex mean + basis)
def sequence_to_vector(sequence, aaindex, aaindex_keys):
    aa_features = []
    for aa in sequence:
        vals = [aaindex[k].get(aa, 0.0) for k in aaindex_keys]
        aa_features.append(vals)
    mean_aaindex = np.mean(aa_features, axis=0) if aa_features else np.zeros(len(aaindex_keys), dtype=np.float32)

    basis = compute_basis_features(sequence)

    return np.concatenate([
        mean_aaindex,
        np.array(basis, dtype=np.float32),
    ])

# ---------------------------
# LMDB index: sequence → FIFO list of stability_score
def build_lmdb_index(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=2048)
    idx = {}
    n = 0
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            try:
                entry = pickle.loads(value)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            seq = entry.get('primary')
            target_list = entry.get('stability_score')  # <-- stability
            if seq is None or target_list is None:
                continue
            y = float(target_list[0])
            idx.setdefault(seq, []).append(y)
            n += 1
    print(f"[LMDB] Indexed {n} entries from {os.path.basename(lmdb_path)} "
          f"({len(idx)} unique sequences)")
    return idx

# ---------------------------
# Align one split to FASTA order
def align_split(fasta_path, lmdb_path, aaindex, aaindex_keys):
    print(f"\n[Align] {os.path.basename(fasta_path)}  ←  {os.path.basename(lmdb_path)}")
    lmdb_idx = build_lmdb_index(lmdb_path)

    with open(fasta_path, 'r') as f:
        fasta_seqs = [ln.strip() for ln in f if ln and not ln.startswith('>')]
    N = len(fasta_seqs)
    print(f"[FASTA] {N} sequences")

    X_list, y_list = [], []
    missing = 0
    seq_lengths = []

    for seq in fasta_seqs:
        lst = lmdb_idx.get(seq)
        if not lst:
            missing += 1
            continue
        y = lst.pop(0)   # FIFO for duplicates
        vec = sequence_to_vector(seq, aaindex, aaindex_keys)
        X_list.append(vec); y_list.append(y)
        seq_lengths.append(len(seq))
        if len(lst) == 0:
            lmdb_idx.pop(seq, None)

    if missing > 0:
        print(f"[WARN] {missing} FASTA sequences not found in LMDB index.")
    leftovers = sum(len(v) for v in lmdb_idx.values())
    if leftovers > 0:
        print(f"[INFO] {leftovers} LMDB entries not used (not present in FASTA order).")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    print(f"[OK] Aligned shapes: X={X.shape}, y={y.shape}")

    min_len = int(np.min(seq_lengths)) if len(seq_lengths) > 0 else 0
    max_len = int(np.max(seq_lengths)) if len(seq_lengths) > 0 else 0

    return X, y, min_len, max_len

# ---------------------------
# Main
if __name__ == '__main__':
    aaindex, aaindex_keys = parse_aaindex1(AAINDEX_PATH)

    # Names (we keep ALL columns; no subset drop)
    feature_names_all = aaindex_keys + basis_names
    with open('feature_names_all.txt', 'w') as f:
        for n in feature_names_all:
            f.write(n + '\n')
    with open('feature_names.txt', 'w') as f:
        for n in feature_names_all:
            f.write(n + '\n')
    print(f"[OUT] feature_names_all.txt ({len(feature_names_all)} names)")

    # Align splits
    X_tr_raw, y_tr, tr_min_len, tr_max_len = align_split(TRAIN_FASTA, TRAIN_LMDB, aaindex, aaindex_keys)
    X_va_raw, y_va, va_min_len, va_max_len = align_split(VALID_FASTA, VALID_LMDB, aaindex, aaindex_keys)
    X_te_raw, y_te, te_min_len, te_max_len = align_split(TEST_FASTA,  TEST_LMDB,  aaindex, aaindex_keys)

    # Save aligned raw
    np.save('X_train_aligned.npy', X_tr_raw); np.save('y_train_aligned.npy', y_tr)
    np.save('X_valid_aligned.npy', X_va_raw); np.save('y_valid_aligned.npy', y_va)
    np.save('X_test_aligned.npy',  X_te_raw); np.save('y_test_aligned.npy',  y_te)
    print("[OUT] Saved aligned arrays.")

    # Standardize (all columns)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)
    X_te = scaler.transform(X_te_raw)

    np.save('X_train_std.npy', X_tr)
    np.save('X_valid_std.npy', X_va)
    np.save('X_test_std.npy',  X_te)
    with open('scaler.pkl', 'wb') as f:
        pkl.dump(scaler, f)
    print("[OUT] Saved standardized arrays + scaler.pkl")

    # Sanity prints
    print("\n[Sanity] Targets:")
    for name, y in [('train', y_tr), ('valid', y_va), ('test', y_te)]:
        print(f"  {name}: n={len(y):5d}  mean={np.mean(y):.6f}  std={np.std(y):.6f}")

    print("\n[Sanity] Sequence lengths:")
    for name, mn, mx in [('train', tr_min_len, tr_max_len),
                         ('valid', va_min_len, va_max_len),
                         ('test',  te_min_len, te_max_len)]:
        print(f"  {name}: min_len={mn}  max_len={mx}")

    print("\n✅ Done. Training should point at X_*_std.npy.")



