import os
import json
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel

from src_emo.critic_supervised_dataset_v2 import CriticSupervisedDatasetV2


@dataclass
class CriticTrainConfig:
    data_dir: str = "/data1/s3905993/ECR-main/src_emo/data/redial_gen/scored_datasets"
    train_parts: List[str] = (
        "llama2_scored_ultra_fast_merged_1_3_part_1.jsonl",
        "llama2_scored_ultra_fast_merged_1_3_part_2.jsonl",
        "llama2_scored_ultra_fast_merged_1_3_part_3.jsonl",
        "llama2_scored_ultra_fast_merged_1_3_part_4.jsonl",
        "llama2_scored_ultra_fast_merged_1_3_part_5.jsonl",
    )
    val_part: str = "llama2_scored_ultra_fast_merged_1_3_part_6.jsonl"
    # Use local cached snapshot to support offline training
    model_name: str = "/data1/s3905993/ECR-main/local_models/roberta-base"
    batch_size: int = 8
    epochs: int = 10
    lr: float = 3e-5
    max_len: int = 384
    weight_decay: float = 0.01
    dropout: float = 0.35
    grad_clip_norm: float = 1.0
    warmup_ratio: float = 0.06
    save_path: str = "critic_roberta_best_v2.pth"
    eval_log_path: str = "critic_roberta_eval_v2.json"


class CriticRobertaMultiHeadV2(nn.Module):
    def __init__(self, model_name: str = "/data1/s3905993/ECR-main/local_models/roberta-base", num_outputs: int = 5, dropout: float = 0.2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.roberta.config.hidden_size
        self.pooler = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, num_outputs)
        self.activation = nn.Sigmoid()  # outputs in [0,1] per dimension

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over valid tokens to stabilize training
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom
        pooled = self.pooler(pooled)
        logits = self.head(pooled)
        return self.activation(logits)  # normalized scores


def build_dataloaders(cfg: CriticTrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
    train_datasets = [
        CriticSupervisedDatasetV2(os.path.join(cfg.data_dir, part), tokenizer, max_length=cfg.max_len)
        for part in cfg.train_parts
    ]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = CriticSupervisedDatasetV2(os.path.join(cfg.data_dir, cfg.val_part), tokenizer, max_length=cfg.max_len)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def evaluate(model, val_loader, device):
    loss_fn = nn.MSELoss()
    model.eval()
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) / 9.0  # normalize targets to [0,1]
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)
            total += input_ids.size(0)
    avg_loss = total_loss / max(total, 1)
    return avg_loss


def main():
    # Allow simple env overrides for ablation runs
    cfg = CriticTrainConfig()
    cfg.data_dir = os.getenv('CRITIC_DATA_DIR', cfg.data_dir)
    env_train = os.getenv('CRITIC_TRAIN_PARTS')
    if env_train:
        cfg.train_parts = tuple(x for x in env_train.split(',') if x)
    cfg.val_part = os.getenv('CRITIC_VAL_PART', cfg.val_part)
    cfg.model_name = os.getenv('CRITIC_MODEL_NAME', cfg.model_name)
    cfg.batch_size = int(os.getenv('CRITIC_BATCH_SIZE', cfg.batch_size))
    cfg.epochs = int(os.getenv('CRITIC_EPOCHS', cfg.epochs))
    cfg.lr = float(os.getenv('CRITIC_LR', cfg.lr))
    cfg.max_len = int(os.getenv('CRITIC_MAX_LEN', cfg.max_len))
    cfg.dropout = float(os.getenv('CRITIC_DROPOUT', cfg.dropout))
    cfg.warmup_ratio = float(os.getenv('CRITIC_WARMUP_RATIO', cfg.warmup_ratio))
    cfg.save_path = os.getenv('CRITIC_SAVE_PATH', cfg.save_path)
    cfg.eval_log_path = os.getenv('CRITIC_EVAL_LOG', cfg.eval_log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    model = CriticRobertaMultiHeadV2(cfg.model_name, num_outputs=5, dropout=cfg.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    # Scheduler with linear warmup and decay
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float('inf')
    history = []
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) / 9.0  # normalize targets to [0,1]
            # label jitter (regularization)
            noise = torch.randn_like(labels) * 0.02
            labels_j = (labels + noise).clamp(0.0, 1.0)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels_j)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += loss.item() * input_ids.size(0)
            seen += input_ids.size(0)

        avg_train = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": val_loss})
        print(f"Epoch {epoch+1} - Train {avg_train:.4f} - Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.save_path)
            print("  (Best model saved)")

    with open(cfg.eval_log_path, "w") as f:
        json.dump({"best_val": best_val, "history": history}, f, indent=2)
    print(f"Training complete. Best Val: {best_val:.4f}; Saved to {cfg.save_path}")


if __name__ == "__main__":
    main()


