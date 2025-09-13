#!/bin/bash

# To run: sbatch score_ultra_fast_merged_1_3.slurm <PART_NUM>
# Each run processes one split: part 1, part 2, ..., part 10

#SBATCH --partition=gpu-medium
#SBATCH --gpus=1
#SBATCH --job-name=llama2_score_ultra_1_3_part_1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/llama2_score_ultra_1_3_part_1_%A_%x.out
#SBATCH --error=slurm_outputs/llama2_score_ultra_1_3_part_1_%A_%x.err

set -e

module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/transformers"
mkdir -p slurm_outputs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed
export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"
export PYTHONPATH="/data1/s3905993/conda_envs/ecrhmas_fixed/lib/python3.10/site-packages:$PYTHONPATH"

cd /data1/s3905993/ECR-main

# Get part number from command line
PART_NUM=$1
if [ -z "$PART_NUM" ]; then
  echo "Error: Must provide part number (1-10) as argument."
  exit 1
fi

INPUT_FILE="src_emo/data/redial_gen/train_scorer_1_3_part_${PART_NUM}.jsonl"
OUTPUT_FILE="llama2_scored_ultra_fast_merged_1_3_part_${PART_NUM}.jsonl"

echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "BitsAndBytes available: $(python -c 'import bitsandbytes; print(bitsandbytes.__version__)')"
echo "Working dir: $(pwd)"
nvidia-smi

echo "=== LLAMA2 ULTRA-FAST SINGLE PART SCORING JOB ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Part: $PART_NUM"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Start time: $(date)"
echo ""

echo "Starting ULTRA-FAST scoring for $INPUT_FILE..."
echo "This version uses 4-bit quantization and should be 5-10x faster!"
python -u src_emo/scoring/llama2_score_responses_ultra_fast.py --input $INPUT_FILE --output $OUTPUT_FILE

echo ""
echo "Finished $INPUT_FILE at: $(date)"
echo "Check output file: $OUTPUT_FILE"
echo "--------------------------------------" 