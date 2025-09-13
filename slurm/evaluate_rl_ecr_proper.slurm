#!/bin/bash
#SBATCH --job-name=rl_ecr_proper_eval
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data1/s3905993/slurm_outputs/rl_ecr_proper_eval_%j.out
#SBATCH --error=/data1/s3905993/slurm_outputs/rl_ecr_proper_eval_%j.err

# Activate conda environment
source /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed
export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"

# Verify environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
python -c "import numpy; print('Numpy available: YES')"

echo "=== RL-Enhanced ECR Proper Evaluation Configuration ==="
echo "Model: RL-enhanced Llama2-7B-Chat"
echo "Evaluation: Recommendation metrics (AUC, RT@K, R@K) + Subjective LLM scoring"
echo "Dataset: Redial test dataset"
echo "Output: results/rl_ecr_evaluation_proper.json"
echo "======================================================"

# Run the RL-enhanced ECR evaluation
python src_emo/evaluation/evaluate_rl_ecr_proper.py \
    --base_model /data1/s3905993/ECRHMAS/src/models/llama2_chat \
    --rl_model models/rl_enhanced_llama2_final \
    --test_file src_emo/data/redial/test_data_processed.jsonl \
    --num_samples 100 \
    --output_file results/rl_ecr_evaluation_proper.json

echo "RL-enhanced ECR proper evaluation completed!" 