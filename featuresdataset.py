#featuresdataset.py ------------------------------------------------------------------------------------------------------------------
"""
Dataset and collate utilities for stability prediction using only global sequence features.

This module:
- Loads protein sequences from the features_CSV.py or features_FASTA.py file.
- Loads aligned, standardized per sequence features (from features[].py).
- Loads scalar experimental stability labels.
- Tokenizes sequences with a Hugging Face ESM2 tokenizer.
- Returns, for each sample:
  - input_ids (ESM2 tokens)
  - features (global physicochemical feature vector)
  - targets (stability value)
- Provides make_collate_fn to:
  - Pad variable length token sequences.
  - Build an attention mask.
  - Batch features and labels for training and evaluation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer  # ESM2 tokenizer

class StabilityWithFeaturesDataset(Dataset):
    """
    Loads sequences, standardized per-sequence features, labels.
    """
    def __init__(self, fasta_path, features_path, labels_path,
                 max_length=512,
                 esm_model_name: str = 'facebook/esm2_t33_650M_UR50D'):
        # Hugging Face ESM2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name, do_lower_case=False)
        self.max_length = max_length

        with open(fasta_path,'r') as f:
            self.sequences=[ln.strip() for ln in f if ln and not ln.startswith('>')]

        # Load features; tolerate object dtype by replacing with a numeric dummy column
        arr = np.load(features_path)
        if arr.dtype == object:
            raise ValueError(f"Features file {features_path} has dtype=object, this is unexpected.")
        self.features = arr.astype(np.float32)

        #load labels
        self.labels  =np.load(labels_path).astype(np.float32)

        #Make sure lengths match
        assert len(self.sequences)==len(self.features)==len(self.labels), \
            f"Mismatch: {len(self.sequences)} seqs, {len(self.features)} feats, {len(self.labels)} labels"


    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq=self.sequences[idx]

        # ESM2 encoding (adds <cls> and <eos>; truncates to max_length)
        token_ids=self.tokenizer.encode(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )

        item={'input_ids':torch.tensor(token_ids,dtype=torch.long),
              'features': torch.from_numpy(self.features[idx]),
              'targets':  torch.tensor(self.labels[idx],dtype=torch.float32)}

        return item

def make_collate_fn(pad_id: int = 0):
    """
    Collate function that pads variable-length ESM2 token sequences,
    builds an attention mask, and batches the global features and labels.
    """
    def collate(batch):
        seqs=[b['input_ids'] for b in batch]
        feats=torch.stack([b['features'] for b in batch])
        targs=torch.stack([b['targets'] for b in batch])

        L=max(len(s) for s in seqs)
        input_ids=torch.full((len(seqs),L), pad_id, dtype=torch.long)
        input_mask=torch.zeros((len(seqs),L), dtype=torch.float32)
        for i,s in enumerate(seqs):
            input_ids[i,:len(s)]=s
            input_mask[i,:len(s)]=1.0

        out={'input_ids':input_ids,'input_mask':input_mask,'features':feats,'targets':targs}

        return out
    return collate
