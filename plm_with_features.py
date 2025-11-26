#PLM_with_features ------------------------------------------------
"""
PLM backbone with fused global and per token features for protein stability regression.

This module:
- Wraps a Hugging Face ESM2 style model as the protein language model (PLM) backbone.
- Optionally injects LoRA adapters into attention projections for parameter efficient fine tuning.
- Provides utilities to inspect the PLM and infer appropriate LoRA target module names.
- Defines CrossFusionBlock to fuse per token side information into PLM token embeddings.
- Defines PLM_With_Features, which:
  - encodes sequences with the PLM,
  - optionally cross fuses per token features,
  - pools token embeddings with a learned attention,
  - concatenates pooled PLM embeddings with global features,
  - and predicts a scalar regression target (for example stability).
"""

import torch
import torch.nn as nn
from transformers import AutoModel  # ESM2 backbone

# PEFT / LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

#Apply a sigmoid to logit scores.
def sigmoid_from_logit(x):
    return torch.sigmoid(x)


def _linear_module_last_names(model: nn.Module):
    """Return the set of last-component names for all Linear submodules."""
    names = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            names.add(name.split(".")[-1])
    return names


def _guess_lora_targets(hf_model: nn.Module):
    """
    Inspect the backbone and pick appropriate LoRA target module names across
    common ESM2 variants / Transformers versions.

    Returns a list of substrings that PEFT will match against module names.
    """
    last_names = _linear_module_last_names(hf_model)

    # Newer ESM releases (HF) — explicit projections
    if {"q_proj", "k_proj", "v_proj"} & last_names:
        targets = ["q_proj", "k_proj", "v_proj"]
        if "out_proj" in last_names:
            targets.append("out_proj")
        return targets

    # PLM-style naming found in some ports / forks
    if {"query", "key", "value"} & last_names:
        targets = ["query", "key", "value"]
        if "dense" in last_names:
            targets.append("dense")
        return targets

    # Fused projections (single Linear for QKV)
    for cand in ["qkv", "qkv_proj", "in_proj", "c_attn", "Wqkv"]:
        if cand in last_names:
            outs = []
            if "out_proj" in last_names:
                outs.append("out_proj")
            if "dense" in last_names:
                outs.append("dense")
            return [cand] + outs

    # Fall back: any Linear living inside attention blocks
    attn_targets = set()
    for name, m in hf_model.named_modules():
        if isinstance(m, nn.Linear) and (".attention." in name or ".attn" in name):
            attn_targets.add(name.split(".")[-1])
    if attn_targets:
        return sorted(attn_targets)

    raise ValueError(
        "Could not infer LoRA target modules in ESM backbone. "
        "Please pass explicit lora_target_modules."
    )

#Check whether at least one of the requested LoRA target names exists.
def _targets_exist(hf_model: nn.Module, targets):
    last_names = _linear_module_last_names(hf_model)
    return any(t in last_names for t in targets)



class PLM_With_Features(nn.Module):
    """
    Protein language model backbone with global per sequence features.

    The model:
    - Loads an ESM2 style backbone from Hugging Face as self.PLM.
    - Optionally attaches LoRA adapters to attention projections for PEFT.
    - Applies a learned attention over PLM token embeddings to obtain a pooled
      sequence vector.
    - Projects global per sequence features to the same hidden size.
    - Concatenates pooled PLM embedding and feature embedding, then predicts a
      scalar regression output with an MLP head.
    """
    def __init__(self, feature_dim: int,
                 PLM_model_name: str = 'facebook/esm2_t33_650M_UR50D',
                 dropout: float = 0.1,
                 # LoRA knobs
                 use_lora: bool = True,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 lora_target_modules = ("q_proj", "k_proj", "v_proj", "out_proj")):
        super().__init__()
        # HF ESM2 backbone
        self.PLM = AutoModel.from_pretrained(PLM_model_name)

        # Inject LoRA adapters on attention projections if requested.
        if use_lora:
            if not _HAS_PEFT:
                raise ImportError(
                    "peft is required for LoRA. Install with `pip install peft` "
                    "or set use_lora=False."
                )

            # Resolve TaskType robustly across PEFT versions
            try:
                task_type = TaskType.FEATURE_EXTRACTION
            except Exception:
                # acceptable fallback for encoders
                task_type = TaskType.TOKEN_CLS

            # Use provided targets if they exist, otherwise auto detect
            targets = list(lora_target_modules) if lora_target_modules else []
            if not targets or not _targets_exist(self.PLM, targets):
                targets = _guess_lora_targets(self.PLM)

            lora_cfg = LoraConfig(
                task_type=task_type,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=targets,
            )
            self.PLM = get_peft_model(self.PLM, lora_cfg)
            # Uncomment for a one line summary during bring up:
            # self.PLM.print_trainable_parameters()

        hidden = self.PLM.config.hidden_size

        # Attention over token embeddings
        self.attn = nn.Linear(hidden, 1)

        self.seq_dropout = nn.Dropout(dropout)

        # Project global features to PLM hidden size
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Regression head on concatenated [pooled_seq, feature_emb]
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

        self.loss_fn = nn.MSELoss()


    @staticmethod
    def _masked_softmax(logits, mask, dim=-1):
        """
        Apply softmax to logits while masking out padding positions.
        Arguments:
            logits: unnormalized attention scores, shape (B, L).
            mask: float or bool mask where 1 marks valid tokens and 0 marks pad.
            dim: dimension to normalize over, default is last.
        Returns:
            Attention weights with softmax over non masked positions.
        """
        logits = logits.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(logits, dim=dim)

    def forward(self, input_ids, input_mask, features, targets=None, per_token_feats=None):
        """
        Run the PLM, fuse features, and compute regression loss.
        """
        # HF models use attention_mask
        outputs = self.PLM(input_ids=input_ids, attention_mask=input_mask)
        seq_out = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]  # (B, L, H)

        # Learned attention over tokens
        attn_logits = self.attn(seq_out).squeeze(-1)  # (B, L)
        attn_weights = self._masked_softmax(attn_logits, input_mask)
        pooled = torch.bmm(attn_weights.unsqueeze(1), seq_out).squeeze(1)
        pooled = self.seq_dropout(pooled)

        # Fuse global features
        feat_emb = self.feature_proj(features)
        preds = self.regressor(torch.cat([pooled, feat_emb], dim=-1)).squeeze(-1)

        if targets is not None:
            loss = self.loss_fn(preds, targets)
            return loss, preds
        return preds


