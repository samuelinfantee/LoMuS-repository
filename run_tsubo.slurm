#!/bin/bash
#SBATCH --job-name=tsubo_int
#SBATCH --partition=Quick
#SBATCH --gres=gpu:A100:1
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/home/s/sinfante/protstab
#SBATCH --output=results/tsuboyama/%x_%A_%a.out
#SBATCH --error=results/tsuboyama/%x_%A_%a.err
#SBATCH --array=0

set -euo pipefail

# --- env & caches ---
export TMPDIR=/data/sinfante/tmp
export CONDA_PKGS_DIRS=/data/sinfante/conda_pkgs
export HF_HOME=/data/sinfante/hf_home
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$HF_HOME"

PROJECT="$HOME/protstab"
DATAROOT="/data/sinfante/protstab/data"
mkdir -p "$PROJECT" "$DATAROOT"
[ -e "$PROJECT/data" ] || ln -s "$DATAROOT" "$PROJECT/data"

PYBIN="/data/sinfante/envs/tape/bin/python"

cd "$PROJECT"

# ensure SLURM log dir exists (SBATCH paths are relative to submit dir)
mkdir -p results/tsuboyama

# --- dataset/protein bucket ---
PROTEIN="tsub_mega"
ROOT="data/dms_one"

# sanity check: features exist
for f in train_seqs.txt valid_seqs.txt test_seqs.txt X_train_std.npy X_valid_std.npy X_test_std.npy y_train_aligned.npy y_valid_aligned.npy y_test_aligned.npy; do
  if [[ ! -s "${ROOT}/${PROTEIN}/$f" ]]; then
    echo "Missing ${ROOT}/${PROTEIN}/$f"
    echo "Run first: python features.py --root ${ROOT} --protein ${PROTEIN}"
    exit 2
  fi
done

# --- LoRA sweep ---
declare -a LORA_R=( 9 )
LORA_ALPHA=16
LORA_TARGETS="q_proj,k_proj,v_proj,out_proj"

IDX=${SLURM_ARRAY_TASK_ID}
R=${LORA_R[$IDX]}

# fixed LRs (edit if you want)
LR_PLM="3.5e-4"
LR_HEAD="2.5e-4"

TAG="tsubo_integrated_gradients"

# --- tsuboyama output folder ---
REV_BASE="${PROJECT}/results/tsuboyama"
RUN_DIR="${REV_BASE}/${TAG}"
LOGDIR="${RUN_DIR}/logs"
CKPTDIR="${RUN_DIR}/checkpoints"
IGDIR="${RUN_DIR}/integrated_gradients"
mkdir -p "$LOGDIR" "$CKPTDIR" "$IGDIR"

BEST_CKPT="${CKPTDIR}/best__${TAG}.pt"

RUN_TXT="${RUN_DIR}/run.txt"
GPU_CSV="${RUN_DIR}/gpu_${SLURM_JOB_ID}_${IDX}.csv"

# Write header
{
  echo "=============================="
  echo "RUN TAG:        ${TAG}"
  echo "DATE:           $(date)"
  echo "JOB:            ${SLURM_JOB_NAME:-NA}"
  echo "JOB_ID:         ${SLURM_JOB_ID:-NA}"
  echo "ARRAY_TASK_ID:  ${SLURM_ARRAY_TASK_ID:-NA}"
  echo "NODE:           $(hostname)"
  echo "USER:           ${USER}"
  echo "PROJECT:        ${PROJECT}"
  echo "ROOT:           ${ROOT}"
  echo "PROTEIN:        ${PROTEIN}"
  echo ""
  echo "LoRA:           use_lora=true freeze_backbone=true"
  echo "LoRA r:         ${R}"
  echo "LoRA alpha:     ${LORA_ALPHA}"
  echo "LoRA targets:   ${LORA_TARGETS}"
  echo ""
  echo "LR plm:         ${LR_PLM}"
  echo "LR head:        ${LR_HEAD}"
  echo "WD plm:         0.01"
  echo "WD head:        0.01"
  echo ""
  echo "RUN_DIR:        ${RUN_DIR}"
  echo "LOGDIR:         ${LOGDIR}"
  echo "CKPTDIR:        ${CKPTDIR}"
  echo "IGDIR:          ${IGDIR}"
  echo "BEST_CKPT:      ${BEST_CKPT}"
  echo "RUN_TXT:        ${RUN_TXT}"
  echo "GPU_CSV:        ${GPU_CSV}"
  echo "=============================="
  echo ""
  echo "=== nvidia-smi (start) ==="
  nvidia-smi || true
  echo ""
  echo "=== torch sanity ==="
  $PYBIN - << 'PY'
import torch
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
  echo ""
} | tee "$RUN_TXT"

# background GPU monitoring (CSV)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 30 > "$GPU_CSV" &
NSMI_PID=$!

# --- train ---
echo "=== TRAIN COMMAND ===" | tee -a "$RUN_TXT"
echo "$PYBIN train_with_features.py --root $ROOT --protein $PROTEIN --run_tag $TAG ..." | tee -a "$RUN_TXT"
echo "" | tee -a "$RUN_TXT"

$PYBIN train_with_features.py \
  --root "$ROOT" \
  --protein "$PROTEIN" \
  --run_tag "$TAG" \
  --epochs 10 \
  --use_lora \
  --freeze_backbone \
  --lora_r "$R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_targets "$LORA_TARGETS" \
  --lr_plm "$LR_PLM" \
  --lr_head "$LR_HEAD" \
  --wd_plm 0.01 \
  --wd_head 0.01 \
  --output_dir "$CKPTDIR" \
  --log_dir "$LOGDIR" \
  --bench_infer \
  2>&1 | tee -a "$RUN_TXT"

echo "" | tee -a "$RUN_TXT"
echo "=== TEST COMMAND ===" | tee -a "$RUN_TXT"
echo "$PYBIN test.py --root $ROOT --protein $PROTEIN --run_tag $TAG ..." | tee -a "$RUN_TXT"
echo "" | tee -a "$RUN_TXT"

$PYBIN test.py \
  --root "$ROOT" \
  --protein "$PROTEIN" \
  --run_tag "$TAG" \
  --use_lora \
  --freeze_backbone \
  --lora_r "$R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_targets "$LORA_TARGETS" \
  --log_dir "$LOGDIR" \
  2>&1 | tee -a "$RUN_TXT"

echo "" | tee -a "$RUN_TXT"
echo "=== INTEGRATED GRADIENTS COMMAND ===" | tee -a "$RUN_TXT"
echo "$PYBIN integrated_gradients.py --root $ROOT --protein $PROTEIN --split test --checkpoint $BEST_CKPT --run_tag ${TAG}_ig ..." | tee -a "$RUN_TXT"
echo "" | tee -a "$RUN_TXT"

if [[ ! -f "$BEST_CKPT" ]]; then
  echo "Missing checkpoint: $BEST_CKPT" | tee -a "$RUN_TXT"
  exit 3
fi

$PYBIN integrated_gradients.py \
  --root "$ROOT" \
  --protein "$PROTEIN" \
  --split test \
  --checkpoint "$BEST_CKPT" \
  --run_tag "${TAG}_ig" \
  --use_lora \
  --freeze_backbone \
  --lora_r "$R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_targets "$LORA_TARGETS" \
  --output_dir "$IGDIR" \
  2>&1 | tee -a "$RUN_TXT"

# stop GPU logger
kill $NSMI_PID || true

echo "" | tee -a "$RUN_TXT"
echo "=== Parsed metrics summary (from JSON logs) ===" | tee -a "$RUN_TXT"

TAG_ENV="$TAG" LOGDIR_ENV="$LOGDIR" $PYBIN - << 'PY' 2>&1 | tee -a "$RUN_TXT"

import os, json
tag = os.environ["TAG_ENV"]
logdir = os.environ["LOGDIR_ENV"]

train_jsonl = os.path.join(logdir, f"{tag}.jsonl")
test_json   = os.path.join(logdir, f"test__{tag}.json")

def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

events = load_jsonl(train_jsonl)

# best val spearman across epochs
best = None
for e in events:
    if e.get("event") == "epoch_end":
        sp = e.get("spearman", None)
        if sp is None:
            continue
        if (best is None) or (sp > best["spearman"]):
            best = e

run_start = next((e for e in events if e.get("event") == "run_start"), None)
run_end   = next((e for e in events if e.get("event") == "run_end"), None)

print(f"train_jsonl: {train_jsonl if os.path.exists(train_jsonl) else 'MISSING'}")
print(f"test_json:   {test_json   if os.path.exists(test_json)   else 'MISSING'}")
print("")

if run_start:
    keys = [
        "protein","lm_name","epochs","batch_size","grad_accum",
        "use_lora","freeze_backbone","lora_r","lora_alpha","lora_targets_used",
        "lr_plm","lr_head","wd_plm","wd_head",
        "total_params","trainable_params"
    ]
    print("Run config:")
    for k in keys:
        if k in run_start:
            print(f"  {k}: {run_start[k]}")
    print("")

if best:
    print("Best validation epoch:")
    print(f"  epoch: {best.get('epoch')}")
    print(f"  spearman: {best.get('spearman')}")
    print(f"  train_seconds: {best.get('train_seconds')}")
    print(f"  train_seq_per_s: {best.get('train_seq_per_s')}")
    print(f"  train_tok_per_s: {best.get('train_tok_per_s')}")
    print(f"  optimizer_state_gb: {best.get('optimizer_state_gb')}")
    print(f"  gpu_peak_alloc_gb: {((best.get('gpu_mem_gb') or {}).get('peak_alloc_gb'))}")
    print(f"  gpu_peak_reserved_gb: {((best.get('gpu_mem_gb') or {}).get('peak_reserved_gb'))}")
    print("")

if run_end:
    print("Run end:")
    print(f"  best_spearman: {run_end.get('best_spearman')}")
    print(f"  checkpoint: {run_end.get('checkpoint')}")
    infer = run_end.get("infer_bench")
    if infer:
        print("  infer_bench:")
        for k,v in infer.items():
            print(f"    {k}: {v}")
    print("")

if os.path.exists(test_json):
    with open(test_json, "r") as f:
        tj = json.load(f)
    print("Test metrics:")
    for k in ["test_loss","test_spearman","infer_seq_per_s","infer_tok_per_s"]:
        if k in tj:
            print(f"  {k}: {tj[k]}")
    mem = tj.get("gpu_mem_gb")
    if mem:
        print(f"  gpu_peak_alloc_gb: {mem.get('peak_alloc_gb')}")
        print(f"  gpu_peak_reserved_gb: {mem.get('peak_reserved_gb')}")
PY

echo "" | tee -a "$RUN_TXT"
echo "=== nvidia-smi (end) ===" | tee -a "$RUN_TXT"
nvidia-smi 2>&1 | tee -a "$RUN_TXT"

echo "" | tee -a "$RUN_TXT"
echo "DONE. Outputs:" | tee -a "$RUN_TXT"
echo "  $RUN_TXT" | tee -a "$RUN_TXT"
echo "  $GPU_CSV" | tee -a "$RUN_TXT"
echo "  $LOGDIR"  | tee -a "$RUN_TXT"
echo "  $CKPTDIR" | tee -a "$RUN_TXT"
echo "  $IGDIR"   | tee -a "$RUN_TXT"
