#!/bin/bash
#SBATCH --job-name=ecr_enhanced_critic
#SBATCH --output=slurm_outputs/ecr_enhanced_critic_%j.out
#SBATCH --error=slurm_outputs/ecr_enhanced_critic_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu-short

# Load modules
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Set working directory
cd /data1/s3905993/ECR-main/src_emo

# Create output directory
mkdir -p ecr_enhanced_critic_output

# Run training
python train_ecr_with_enhanced_critic.py \
    --data_path data/redial/train_data_processed_merge.jsonl \
    --output_dir ecr_enhanced_critic_output \
    --policy_model microsoft/DialoGPT-small \
    --critic_model roberta-base \
    --num_epochs 5 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --max_length 512 \
    --save_steps 50 \
    --device cpu

echo "Enhanced Critic Training completed!" 