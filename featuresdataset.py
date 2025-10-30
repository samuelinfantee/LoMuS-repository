#featuresdataset.py ------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer  # ESM2 tokenizer

# AA deltas (used for per-token mutation deltas vs WT)
KD_HYDROPATHY={'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'W':-0.9,'S':-0.8,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
AA_VOLUME={'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
AA_CHARGE={'A':0,'R':+1,'N':0,'D':-1,'C':0,'Q':0,'E':-1,'G':0,'H':+0.1,'I':0,'L':0,'K':+1,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}

class StabilityWithFeaturesDataset(Dataset):
    """
    Loads sequences, standardized per-sequence features, labels.
    Optional per-token features from a reference WT/core geometry:
      Z_tokens[T, 9] aligned to tokens; zeros at <cls>/<eos>.
    """
    def __init__(self, fasta_path, features_path, labels_path,
                 max_length=512,
                 use_token_feats=False,
                 wt_geom_path='WT_geom_stability.npz',
                 wt_seq_path='data/stability/stability_wt.txt',
                 esm_model_name: str = 'facebook/esm2_t33_650M_UR50D'):
        # Hugging Face ESM2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name, do_lower_case=False)
        self.max_length = max_length
        self.use_token_feats = use_token_feats

        with open(fasta_path,'r') as f:
            self.sequences=[ln.strip() for ln in f if ln and not ln.startswith('>')]

        self.features=np.load(features_path).astype(np.float32)
        self.labels  =np.load(labels_path).astype(np.float32)
        assert len(self.sequences)==len(self.features)==len(self.labels), \
            f"Mismatch: {len(self.sequences)} seqs, {len(self.features)} feats, {len(self.labels)} labels"

        self.WT=None; self.WT_SEQ=None
        if use_token_feats and os.path.exists(wt_geom_path):
            d=np.load(wt_geom_path, allow_pickle=True)
            self.WT={'degree':d['degree'].astype(np.float32),
                     'dist_to_chrom':d['dist_to_chrom'].astype(np.float32),  # core distance
                     'plddt':d['plddt'].astype(np.float32),
                     'is_surface':d['is_surface'].astype(np.float32)}
            self.WT_SEQ=str(d['wt_seq'].tolist()) if 'wt_seq' in d else None
        if use_token_feats and os.path.exists(wt_seq_path):
            with open(wt_seq_path,'r') as f:
                s=f.readline().strip()
            if s: self.WT_SEQ=s

        # [deg_norm, dist_norm, invdist, plddt_norm, surface, is_mut, dH/5, dV/200, dQ]
        self._token_feat_dim = 9

    @property
    def token_feat_dim(self): return self._token_feat_dim
    def __len__(self): return len(self.sequences)

    def _per_residue_feats(self, seq: str):
        L=len(seq)
        if (self.WT is None) or (self.WT_SEQ is None):
            return np.zeros((L,self._token_feat_dim),dtype=np.float32)
        L=min(L, len(self.WT_SEQ), len(self.WT['degree']))
        deg=self.WT['degree'][:L]; dcore=self.WT['dist_to_chrom'][:L]; pl=self.WT['plddt'][:L]; surf=self.WT['is_surface'][:L]
        deg_max=max(float(deg.max()),1.0); dist_max=max(float(dcore.max()),1e-6)
        deg_norm=(deg/deg_max).astype(np.float32)
        dist_norm=(dcore/dist_max).astype(np.float32)
        invdist=(1.0/(1.0+dcore)).astype(np.float32)
        plddt_norm=(pl/100.0).astype(np.float32)
        surface=(surf>0.5).astype(np.float32)

        is_mut=np.zeros(L, dtype=np.float32)
        dH=np.zeros(L, dtype=np.float32); dV=np.zeros(L, dtype=np.float32); dQ=np.zeros(L, dtype=np.float32)
        for i in range(L):
            if seq[i]!=self.WT_SEQ[i]:
                is_mut[i]=1.0
                wt,mt=self.WT_SEQ[i],seq[i]
                dH[i]=KD_HYDROPATHY.get(mt,0.0)-KD_HYDROPATHY.get(wt,0.0)
                dV[i]=AA_VOLUME.get(mt,0.0)-AA_VOLUME.get(wt,0.0)
                dQ[i]=AA_CHARGE.get(mt,0.0)-AA_CHARGE.get(wt,0.0)

        dH=(dH/5.0).astype(np.float32); dV=(dV/200.0).astype(np.float32)
        Z=np.stack([deg_norm,dist_norm,invdist,plddt_norm,surface,is_mut,dH,dV,dQ],axis=-1).astype(np.float32)
        return Z

    def __getitem__(self, idx):
        seq=self.sequences[idx]
        # ESM2 encoding (adds <cls> and <eos>; truncates to max_length)
        token_ids=self.tokenizer.encode(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )
        T=len(token_ids)

        item={'input_ids':torch.tensor(token_ids,dtype=torch.long),
              'features': torch.from_numpy(self.features[idx]),
              'targets':  torch.tensor(self.labels[idx],dtype=torch.float32)}

        if self.use_token_feats:
            Z_res = self._per_residue_feats(seq)             # (L,D)
            D=Z_res.shape[-1]
            Z_tok = np.zeros((T,D), dtype=np.float32)
            # reserve <cls> and <eos>
            L_eff = max(0, T-2)
            if L_eff>0 and Z_res.shape[0]>0:
                copy_len = min(L_eff, Z_res.shape[0])
                Z_tok[1:1+copy_len, :] = Z_res[:copy_len, :]
            item['per_token_feats'] = torch.from_numpy(Z_tok)

        return item

def make_collate_fn(pad_id: int = 0):
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

        if 'per_token_feats' in batch[0]:
            D = batch[0]['per_token_feats'].shape[-1]
            Zpad = torch.zeros((len(batch), L, D), dtype=torch.float32)
            for i,b in enumerate(batch):
                z=b['per_token_feats']
                Zpad[i, :min(L, z.shape[0]), :] = z[:min(L, z.shape[0]), :]
            out['per_token_feats']=Zpad

        return out
    return collate




