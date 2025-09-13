# Enhanced Conversational Recommendation (ECR) with Dual-Model Critic

This repository contains the complete implementation of an Enhanced Conversational Recommendation system with a dual-model critic for response quality assessment and reranking.

## 🎯 Project Overview

This project implements a sophisticated conversational recommendation system that:
- Generates movie recommendations through natural conversation
- Uses a dual-model critic (RoBERTa + LLM) for response quality assessment
- Implements reinforcement learning with policy optimization
- Provides comprehensive evaluation metrics and analysis

## 📁 Repository Structure

```
ECR-main/
├── src_emo/                          # Main source code
│   ├── training/                     # Training scripts
│   ├── evaluation/                   # Evaluation scripts
│   ├── scoring/                      # Response scoring scripts
│   ├── knowledge/                    # Knowledge retrieval modules
│   ├── generation/                   # Response generation modules
│   ├── recommend/                    # Recommendation modules
│   ├── rl/                          # Reinforcement learning components
│   └── data/                        # Data processing utilities
├── slurm_scripts/                    # SLURM job scripts
├── scripts/                         # Analysis and utility scripts
├── plan_progress_md_files/          # Documentation and thesis files
└── results/                         # Evaluation results (generated)
```

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**:
   ```bash
   conda create -n ecrhmas_fixed python=3.10
   conda activate ecrhmas_fixed
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch transformers datasets accelerate
   pip install scikit-learn matplotlib seaborn pandas
   pip install rouge-score bert-score
   ```

3. **Data Preparation**:
   - Download ReDial dataset
   - Process and prepare training data
   - Generate movie review embeddings

### 🏗️ Training Pipeline

#### 1. Train the Dual-Model Critic
```bash
# Submit SLURM job for critic training
sbatch slurm_scripts/train_critic_dual_model.slurm
```

#### 2. Train ECR with Enhanced Critic
```bash
# Submit SLURM job for ECR training
sbatch slurm_scripts/train_ecr_with_enhanced_critic.slurm
```

#### 3. Train RL Policy
```bash
# Submit SLURM job for RL training
sbatch slurm_scripts/train_llama2_rl_with_policy.slurm
```

### 📊 Evaluation Pipeline

#### 1. Evaluate ECR Performance
```bash
# Submit evaluation job
sbatch slurm_scripts/evaluate_ecr_proper.slurm
```

#### 2. Evaluate RL-Enhanced ECR
```bash
# Submit RL evaluation job
sbatch slurm_scripts/evaluate_rl_ecr_proper.slurm
```

#### 3. Score Responses with LLMs
```bash
# Score with Llama2
sbatch slurm_scripts/score_ultra_fast_merged_1_3.slurm
```

## 🔧 Key Components

### Dual-Model Critic
- **RoBERTa Component**: Fast, accurate response quality assessment
- **LLM Component**: Comprehensive evaluation with reasoning
- **Ensemble Scoring**: Combines both models for robust evaluation

### Enhanced ECR System
- **Knowledge Retrieval**: Contextual movie information
- **Response Generation**: Natural conversation flow
- **Quality Assessment**: Real-time response evaluation
- **Reranking**: Optimizes recommendation quality

### Reinforcement Learning
- **Policy Optimization**: Improves response generation
- **Reward Shaping**: Balances multiple objectives
- **Experience Replay**: Efficient learning from interactions

## 📈 Results and Evaluation

The system achieves:
- **High-quality recommendations** with contextual relevance
- **Natural conversation flow** with appropriate responses
- **Robust evaluation** through dual-model critic assessment
- **Improved performance** through RL optimization

### Key Metrics
- **BLEU Score**: Measures response fluency
- **ROUGE Score**: Evaluates content relevance
- **BERTScore**: Assesses semantic similarity
- **Human Evaluation**: Subjective quality assessment

## 🛠️ Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/huggingface
```

### Model Configuration
- **Base Model**: Llama2-7B-Chat
- **Critic Model**: RoBERTa-base + Llama2-7B-Chat
- **Training Data**: ReDial + Movie Reviews
- **Evaluation Data**: Test set with human annotations

## 📝 Usage Examples

### Training a New Critic
```python
from src_emo.train_critic_dual_model import train_critic

# Train dual-model critic
train_critic(
    train_data_path="data/train.jsonl",
    model_save_path="models/critic_dual.pth",
    epochs=10,
    batch_size=32
)
```

### Evaluating Responses
```python
from src_emo.evaluation.evaluate_ecr_proper import evaluate_ecr

# Evaluate ECR performance
results = evaluate_ecr(
    model_path="models/ecr_enhanced.pth",
    test_data_path="data/test.jsonl",
    output_path="results/evaluation.json"
)
```

## 🔬 Research and Thesis

This project is part of a comprehensive research study on conversational recommendation systems. The complete thesis documentation is available in `plan_progress_md_files/`.

### Key Contributions
1. **Dual-Model Critic Architecture**: Novel approach to response quality assessment
2. **Enhanced ECR System**: Improved conversational recommendation pipeline
3. **RL Integration**: Policy optimization for better response generation
4. **Comprehensive Evaluation**: Multi-metric assessment framework

## 📚 References

- ReDial Dataset: Conversational Recommendation Dataset
- Llama2: Large Language Model for Generation
- RoBERTa: Robust BERT for Classification
- PPO: Proximal Policy Optimization Algorithm

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:
1. Check the documentation in `plan_progress_md_files/`
2. Review the evaluation results in `results/`
3. Open an issue on GitHub

## 🎉 Acknowledgments

- ReDial dataset creators
- Hugging Face for model implementations
- The research community for foundational work

---

**Note**: This repository contains all necessary code and documentation to reproduce the research results. Large model files and generated data are excluded via `.gitignore` and should be trained/generated locally following the provided instructions.
