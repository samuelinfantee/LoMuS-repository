#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare ProteinGym DMS (substitutions) datasets for per-protein training.

Example:
  python tools/prepare_unirep_dms.py \
    --raw data/proteingym \
    --out data/dms \
    --seed 42 \
    --min_seqs 200
"""

import argparse, json, os, re, sys, zipfile, io
from pathlib import Path
from collections import defaultdict
import random

import pandas as pd

# --------- Utilities ---------

def log(msg):
    print(f"[prepare_unirep_dms] {msg}", flush=True)

MUT_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def parse_mut_string(mut_str):
    """
    Accepts forms like 'A123C' or 'A123C;G45D' (will return multiple).
    Returns list of (pos0, wt, mut) where pos0 is zero-based index.
    """
    if pd.isna(mut_str):
        return []
    muts = []
    for token in str(mut_str).replace(',', ';').split(';'):
        token = token.strip()
        if not token:
            continue
        m = MUT_RE.match(token)
        if not m:
            # Try alternate formats e.g.,  A123C(whatever)
            token = re.sub(r'\(.*?\)', '', token)
            m = MUT_RE.match(token)
        if not m:
            raise ValueError(f"Unrecognized mutation format: '{token}'")
        wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
        muts.append((pos - 1, wt, mt))  # convert to 0-based
    return muts

def apply_mutations(wt_seq, mut_str):
    seq = list(wt_seq)
    for pos0, wt, mt in parse_mut_string(mut_str):
        if pos0 < 0 or pos0 >= len(seq):
            raise ValueError(f"Mutation position out of bounds: {mut_str} for len {len(seq)}")
        if wt != seq[pos0]:
            # Be tolerant: only warn the first mismatch
            # Many DMS CSVs use reference indexing that may differ; we fail hard to avoid silent corruption
            raise ValueError(f"WT mismatch at {pos0+1}: expected {wt} in mutation {mut_str}, found {seq[pos0]}")
        seq[pos0] = mt
    return ''.join(seq)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def standardize_score_col(df):
    # try common names
    candidates = ['DMS_score', 'score', 'fitness', 'Effect', 'mut_score', 'y', 'target']
    for c in df.columns:
        if c in candidates:
            return c
    # fallback: pick the only float-like column if unique
    float_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(float_cols) == 1:
        return float_cols[0]
    raise KeyError("Could not infer score column; please inspect columns: " + ', '.join(df.columns))

def standardize_mut_col(df):
    # common names for substitution string
    candidates = ['mutant', 'mutation', 'Mut', 'mut', 'aa_substitutions', 'variant', 'substitution']
    for c in df.columns:
        if c in candidates:
            return c
    # alternative: columns (pos, wt, mt)
    if all(x in df.columns for x in ['pos', 'wt', 'mt']):
        # synthesize a mutation string
        df = df.copy()
        df['__mut'] = df['wt'].astype(str) + df['pos'].astype(int).astype(str) + df['mt'].astype(str)
        return df, '__mut'
    raise KeyError("Could not infer mutation column; expected something like 'mutant' or (pos, wt, mt).")

def find_wt_for_protein(protein_name, df):
    """
    Try to obtain WT sequence for a dataset:
    - Look for a dedicated WT mapping CSV in the parent directory
    - Look for per-row columns like 'WT_seq', 'wildtype_sequence'
    - Else return None
    """
    # 1) Search sibling files for a global mapping
    parent = Path(df.attrs.get('__source_dir', '.'))
    mapping_candidates = list(parent.glob('*WT*.csv')) + list(parent.glob('*wildtype*.csv')) + list(parent.glob('*WT*.tsv'))
    for mc in mapping_candidates:
        try:
            map_df = pd.read_csv(mc) if mc.suffix.lower()=='.csv' else pd.read_csv(mc, sep='\t')
            cols = {c.lower(): c for c in map_df.columns}
            # Guess keys
            name_col = cols.get('uniprot_id') or cols.get('name') or cols.get('protein') or list(cols.values())[0]
            wt_col = cols.get('wt_seq') or cols.get('wildtype_sequence') or cols.get('sequence') or None
            if wt_col is None:
                continue
            # Try direct match by file stem or by a 'name' present in CSV
            # Normalize keys
            key = protein_name.upper()
            cand = map_df[map_df[name_col].astype(str).str.upper() == key]
            if len(cand) == 1:
                return str(cand.iloc[0][wt_col])
        except Exception:
            pass

    # 2) Per-row WT sequence presence
    for c in ['WT_seq', 'wildtype_sequence', 'wt_sequence', 'sequence_wt']:
        if c in df.columns and df[c].notna().any():
            vals = df[c].dropna().unique()
            if len(vals) == 1:
                return str(vals[0])

    # Not found
    return None

def write_fasta(path: Path, name: str, seq: str):
    with open(path, 'w') as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

def make_splits(n, seed, ratios=(0.8, 0.1, 0.1)):
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = int(ratios[0]*n)
    n_val   = int(ratios[1]*n)
    train = idx[:n_train]
    val   = idx[n_train:n_train+n_val]
    test  = idx[n_train+n_val:]
    return {'train': train, 'val': val, 'test': test}

# --------- Main pipeline ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help="Directory that contains DMS_ProteinGym_substitutions.zip (or an extracted folder)")
    ap.add_argument('--out', required=True, help="Output directory for per-protein folders")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_seqs', type=int, default=200, help="Skip proteins with < min_seqs rows after cleaning")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    # Unzip if needed
    zip_path = raw_dir / 'DMS_ProteinGym_substitutions.zip'
    if zip_path.exists():
        log(f"Found ZIP at {zip_path}. Extracting (idempotent)...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extract only if file doesn't already exist with same size
            for zi in zf.infolist():
                target = raw_dir / zi.filename
                if target.exists() and target.stat().st_size == zi.file_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                zf.extract(zi, raw_dir)

    # Find CSVs
    candidates = list(raw_dir.glob('**/*.csv'))
    # Keep only “substitution” tables (many distributions name them PROTEIN.csv)
    # If a “metadata” CSV sneaks in, we’ll skip it when columns don’t match.
    if not candidates:
        log("No CSV files found. Did you point --raw to the directory that contains the extracted ZIP?")
        sys.exit(1)

    # Track summary
    kept, skipped = [], []

    for csv_path in sorted(candidates):
        # Heuristic: ignore obvious metadata/README files
        lower_name = csv_path.name.lower()
        if 'readme' in lower_name or 'meta' in lower_name or 'wildtype' in lower_name:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            # try TSV
            try:
                df = pd.read_csv(csv_path, sep='\t')
            except Exception as e:
                log(f"Skipping {csv_path.name}: not a valid table ({e})")
                continue

        # Attach source dir for WT mapping attempts
        df.attrs['__source_dir'] = str(csv_path.parent)

        # Infer mutation column
        try:
            mut_col_info = standardize_mut_col(df)
            if isinstance(mut_col_info, tuple):
                df, mut_col = mut_col_info
            else:
                mut_col = mut_col_info
        except KeyError:
            # Not a DMS substitution table
            continue

        # Infer score column
        try:
            score_col = standardize_score_col(df)
        except KeyError:
            log(f"Skipping {csv_path.name}: couldn't infer score column.")
            continue

        # Normalize + reduce to necessary fields
        use_cols = [mut_col, score_col]
        df2 = df[use_cols].rename(columns={mut_col: 'mut', score_col: 'score'}).copy()

        # Deduplicate (avg repeated mutant measurements)
        df2 = (df2
               .dropna(subset=['mut'])
               .groupby('mut', as_index=False)['score'].mean())

        # Filter out invalid mutation strings early
        valid_rows = []
        for _, row in df2.iterrows():
            mut = row['mut']
            try:
                # Will raise if malformed
                parse_mut_string(mut)
                valid_rows.append(True)
            except Exception:
                valid_rows.append(False)
        df2 = df2.loc[valid_rows].reset_index(drop=True)

        # Drop NaN scores
        df2 = df2.dropna(subset=['score'])

        if len(df2) < args.min_seqs:
            skipped.append((csv_path.name, f"< min_seqs ({len(df2)})"))
            continue

        # Determine protein name from filename (strip extension)
        protein = csv_path.stem
        # Some distributions name files like "YAP1_HUMAN_Araya_2012.csv"
        # That’s perfect as a folder name.
        prot_dir = out_dir / protein
        safe_mkdir(prot_dir)

        # Try to find WT sequence and reconstruct sequences
        wt_seq = find_wt_for_protein(protein, df)
        seq_rows = []
        seq_ok = False
        if wt_seq:
            try:
                for _, r in df2.iterrows():
                    seq_rows.append({
                        'seq': apply_mutations(wt_seq, r['mut']),
                        'score': r['score']
                    })
                seq_df = pd.DataFrame(seq_rows)
                seq_ok = True
            except Exception as e:
                log(f"{protein}: could not reconstruct sequences reliably ({e}). Will write mutations-only.")

        # Write outputs
        # 1) wildtype.fa (only if we have WT)
        if wt_seq:
            write_fasta(prot_dir / 'wildtype.fa', protein, wt_seq)

        # 2) mutations.csv
        df2.to_csv(prot_dir / 'mutations.csv', index=False)

        # 3) seqs.csv (if usable)
        if seq_ok:
            seq_df.to_csv(prot_dir / 'seqs.csv', index=False)

        # 4) split.json
        n = len(seq_df) if seq_ok else len(df2)
        splits = make_splits(n, args.seed, (0.8, 0.1, 0.1))
        with open(prot_dir / 'split.json', 'w') as f:
            json.dump(splits, f, indent=2)

        kept.append((protein, len(df2), 'seqs' if seq_ok else 'mutations-only'))

    # Summary
    log(f"Prepared {len(kept)} datasets into {out_dir}")
    if kept:
        for k in kept[:10]:
            log(f"  - {k[0]}: {k[1]} rows ({k[2]})")
        if len(kept) > 10:
            log(f"  ... and {len(kept)-10} more.")
    if skipped:
        log(f"Skipped {len(skipped)} tables:")
        for name, why in skipped[:10]:
            log(f"  - {name}: {why}")
        if len(skipped) > 10:
            log(f"  ... and {len(skipped)-10} more.")

if __name__ == "__main__":
    main()
