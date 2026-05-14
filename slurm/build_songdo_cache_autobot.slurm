#!/bin/bash
#SBATCH --job-name=songdo_cache_autobot
#SBATCH --output=/home/%u/projects/Songdo/logs/%x_%j.out
#SBATCH --error=/home/%u/projects/Songdo/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --account=master

set -euo pipefail

WORKROOT=${WORKROOT:-/home/$USER/projects/Songdo}
LOG_DIR=${LOG_DIR:-$WORKROOT/logs}
SKYDRIVE_ROOT=${SKYDRIVE_ROOT:-$WORKROOT/Skydrive_Songdo}
UNITRAJ_CONFIG_DIR=${UNITRAJ_CONFIG_DIR:-$WORKROOT/UniTraj_Songdo/unitraj/configs}
TRAIN_DATA=${TRAIN_DATA:-/scratch/izar/$USER/songdo_drive/train}
VAL_DATA=${VAL_DATA:-/scratch/izar/$USER/songdo_drive/test}
CACHE_ROOT=${CACHE_ROOT:-/scratch/izar/$USER/songdo_cache/autobot}
VENV_PATH=${VENV_PATH:-/home/$USER/venvs/mtr-izar/bin/activate}

mkdir -p "$LOG_DIR"

module purge
module load gcc/11.3.0
module load python

source "$VENV_PATH"

export PYTHONUNBUFFERED=1

for required_dir in "$SKYDRIVE_ROOT" "$UNITRAJ_CONFIG_DIR" "$TRAIN_DATA" "$VAL_DATA"; do
  if [ ! -d "$required_dir" ]; then
    echo "Missing required directory: $required_dir" >&2
    exit 1
  fi
done

cd "$SKYDRIVE_ROOT"

python -m skydrive.unitraj.unitraj_cache_builder \
  --method autobot \
  --train-data-path "$TRAIN_DATA" \
  --val-data-path "$VAL_DATA" \
  --unitraj-config-dir "$UNITRAJ_CONFIG_DIR" \
  --cache-path "$CACHE_ROOT" \
  --overwrite-cache \
  --num-workers 8
