#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=Quick
#SBATCH --gres=gpu:A100:1
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/sinfante/protstab/logs/%x_%A_%a.out
#SBATCH --error=/data/sinfante/protstab/logs/%x_%A_%a.err
#SBATCH --array=0-0

set -euo pipefail

# --- env & caches ---
export TMPDIR=/data/sinfante/tmp
export CONDA_PKGS_DIRS=/data/sinfante/conda_pkgs
export HF_HOME=/data/sinfante/hf_home
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
# correct syntax (or just delete this line if you prefer):
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$HF_HOME" /data/sinfante/protstab/logs

PROJECT="$HOME/protstab"
DATAROOT="/data/sinfante/protstab/data"
mkdir -p "$PROJECT" "$DATAROOT"
[ -e "$PROJECT/data" ] || ln -s "$DATAROOT" "$PROJECT/data"

PYBIN="/data/sinfante/envs/tape/bin/python"

echo "Node: $(hostname)"
echo "User: $USER"
nvidia-smi || true

$PYBIN - << 'PY'
import torch
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

cd "$PROJECT"

# --- dataset/protein bucket ---
PROTEIN="PIN1_HUMAN_Tsuboyama_2023_1I6C"

# sanity check: features exist (you said features.py ran already)
for f in train_seqs.txt valid_seqs.txt test_seqs.txt X_train_std.npy X_valid_std.npy X_test_std.npy y_train_aligned.npy y_valid_aligned.npy y_test_aligned.npy; do
  if [[ ! -s "data/dms_one/${PROTEIN}/$f" ]]; then
    echo "Missing data/dms_one/${PROTEIN}/$f"
    echo "Run first: python features.py --root data/dms_one --protein ${PROTEIN}"
    exit 2
  fi
done

# --- 5-point LR sweep ---
declare -a PLM_LRS=( 3.5e-4 )
declare -a FUSN_LRS=( 2.5e-4)

IDX=${SLURM_ARRAY_TASK_ID}
LR_PLM=${PLM_LRS[$IDX]}
LR_FUSN=${FUSN_LRS[$IDX]}
TAG="tsub_lrB${LR_PLM}_lrF${LR_FUSN}"

echo "Array index: $IDX"
echo "LRs: PLM=$LR_PLM  fusion=$LR_FUSN"
echo "Tag: $TAG"

# --- train ---
$PYBIN train_with_features_yes_no.py \
  --root data/dms_one \
  --protein "$PROTEIN" \
  --lr_PLM "$LR_PLM" \
  --lr_fusion "$LR_FUSN" \
  --run_tag "$TAG"

# --- test the exact checkpoint ---
$PYBIN test_yes_no.py \
  --root data/dms_one \
  --protein "$PROTEIN" \
  --run_tag "$TAG"
