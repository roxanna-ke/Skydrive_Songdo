#!/bin/bash
#SBATCH --job-name=songdo_cache_mtr
#SBATCH --output=/home/ke/projects/Songdo/logs/%x_%j.out
#SBATCH --error=/home/ke/projects/Songdo/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu

set -euo pipefail

mkdir -p /home/ke/projects/Songdo/logs

module purge
module load gcc/11.3.0
module load python

source /home/ke/venvs/mtr-izar/bin/activate

export PYTHONUNBUFFERED=1

WORKROOT=/home/ke/projects/Songdo
SKYDRIVE_ROOT=$WORKROOT/Skydrive_Songdo
UNITRAJ_CONFIG_DIR=$WORKROOT/UniTraj_Songdo/unitraj/configs
TRAIN_DATA=/scratch/izar/ke/songdo_drive/train
VAL_DATA=/scratch/izar/ke/songdo_drive/test
CACHE_ROOT=/scratch/izar/ke/songdo_cache/mtr

cd "$SKYDRIVE_ROOT"

python -m skydrive.unitraj.unitraj_cache_builder \
  --method MTR \
  --train-data-path "$TRAIN_DATA" \
  --val-data-path "$VAL_DATA" \
  --unitraj-config-dir "$UNITRAJ_CONFIG_DIR" \
  --cache-path "$CACHE_ROOT" \
  --overwrite-cache \
  --num-workers 8
