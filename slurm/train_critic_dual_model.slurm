#!/bin/bash
#SBATCH --job-name=critic_dual_model
#SBATCH --output=logs/critic_dual_model_%j.out
#SBATCH --error=logs/critic_dual_model_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s3905993@liacs.leidenuniv.nl

# Activate conda environment (consistent with other jobs)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Set working directory
cd /data1/s3905993/ECR-main

# Create logs directory
mkdir -p logs

# Print job information
echo "=== CRITIC DUAL-MODEL TRAINING JOB ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "======================================"

# Check if datasets exist
echo "Checking dataset files..."
if [ ! -f "src_emo/data/redial_gen/scored_datasets/critic_train_dual_model.jsonl" ]; then
    echo "ERROR: Training dataset not found!"
    exit 1
fi

if [ ! -f "src_emo/data/redial_gen/scored_datasets/critic_val_dual_model.jsonl" ]; then
    echo "ERROR: Validation dataset not found!"
    exit 1
fi

if [ ! -f "src_emo/data/redial_gen/scored_datasets/critic_full_dual_model.jsonl" ]; then
    echo "ERROR: Full dataset not found!"
    exit 1
fi

echo "All dataset files found!"

# Print dataset statistics
echo "Dataset Statistics:"
echo "Training samples: $(wc -l < src_emo/data/redial_gen/scored_datasets/critic_train_dual_model.jsonl)"
echo "Validation samples: $(wc -l < src_emo/data/redial_gen/scored_datasets/critic_val_dual_model.jsonl)"
echo "Full dataset samples: $(wc -l < src_emo/data/redial_gen/scored_datasets/critic_full_dual_model.jsonl)"

# Run critic training (offline with local cache)
echo "Starting critic training..."
export HF_HOME=/data1/s3905993/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
python -u src_emo/train_critic_dual_model.py \
    --train_data src_emo/data/redial_gen/scored_datasets/critic_train_dual_model.jsonl \
    --val_data src_emo/data/redial_gen/scored_datasets/critic_val_dual_model.jsonl \
    --test_data src_emo/data/redial_gen/scored_datasets/critic_full_dual_model.jsonl \
    --output_dir critic_pretrained_dual_model \
    --model_name roberta-base \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_length 256 \
    --seed 42

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=== TRAINING COMPLETED SUCCESSFULLY ==="
    echo "Model saved to: critic_pretrained_dual_model/"
    echo "Training results saved to: critic_pretrained_dual_model/training_results.json"
    
    # List output files
    echo "Output files:"
    ls -la critic_pretrained_dual_model/
    
    # Show final model size
    if [ -f "critic_pretrained_dual_model/critic_final.pth" ]; then
        echo "Final model size: $(du -h critic_pretrained_dual_model/critic_final.pth | cut -f1)"
    fi
else
    echo "=== TRAINING FAILED ==="
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed!" 