#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=Quick
#SBATCH --gres=gpu:A100:1
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/sinfante/protstab/logs/%x_%A_%a.out
#SBATCH --error=/data/sinfante/protstab/logs/%x_%A_%a.err
#SBATCH --array=0-0   # 5 tasks total, max 5 running concurrently

set -euo pipefail

# --- env & paths (same as yours) ---
export TMPDIR=/data/sinfante/tmp
export CONDA_PKGS_DIRS=/data/sinfante/conda_pkgs
export HF_HOME=/data/sinfante/hf_home
export MPLBACKEND=Agg
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$HF_HOME" /data/sinfante/protstab/logs

PROJECT="$HOME/protstab"
DATAROOT="/data/sinfante/protstab/data"
mkdir -p "$PROJECT" "$DATAROOT"
[ -e "$PROJECT/data" ] || ln -s "$DATAROOT" "$PROJECT/data"

PYBIN="/data/sinfante/envs/tape/bin/python"

echo "Node: $(hostname)"
echo "Python: $PYBIN"

$PYBIN - << 'PY'
import torch
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY

cd "$PROJECT"

# --- choose your protein ---
PROTEIN="PIN1_HUMAN_Tsuboyama_2023_1I6C"

# --- define 5 LR pairs (PLM_lr, fusion_lr) ---
# You can edit these to whatever you want.
declare -a PLM_LRS=( 3.5e-4)
declare -a FUSN_LRS=( 2e-4)

IDX=${SLURM_ARRAY_TASK_ID}
LR_PLM=${PLM_LRS[$IDX]}
LR_FUSN=${FUSN_LRS[$IDX]}

# optional: distinct tag per run so checkpoints don't collide
TAG="lrB${LR_PLM}_lrF${LR_FUSN}"

echo "Run index: $IDX"
echo "LRs: PLM=$LR_PLM fusion=$LR_FUSN"
echo "Tag: $TAG"

# --- train ---
$PYBIN train_with_features.py \
  --root data/dms_one \
  --protein "$PROTEIN" \
  --lr_PLM "$LR_PLM" \
  --lr_fusion "$LR_FUSN" \
  --run_tag "$TAG"

# --- test the exact checkpoint produced by this run ---
$PYBIN test.py \
  --root data/dms_one \
  --protein "$PROTEIN" \
  --run_tag "$TAG"




