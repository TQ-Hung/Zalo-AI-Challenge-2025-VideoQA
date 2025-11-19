# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/train_advanced.py
import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import FeatureVideoQADataset, collate_fn
from model_improved import ImprovedVideoQA
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

# ---------- Advanced Config ----------
MODEL_TEXT = "vinai/phobert-base"
BATCH_SIZE = 16  # Increased batch size
MAX_LEN = 128    # Increased sequence length
LR = 3e-5
EPOCHS = 30
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "/kaggle/working/checkpoints_advanced"
VALID_SPLIT = 0.1
SEED = 42

USE_FP16 = True
ACCUM_STEPS = 1
WARMUP_STEPS = 100
CLIP_NORM = 1.0
NUM_WORKERS = 4

# Advanced training
APPEARANCE_DIR = "/kaggle/working/features_v2/appearance"
MOTION_DIR = "/kaggle/working/features_v2/motion"
DATA_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"

# Focal Loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_advanced():
    seed_everything()
    
    # Initialize wandb for experiment tracking
    wandb.init(project="zalo-videoqa-advanced", config={
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    })
    
    # Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT)
    full_ds = FeatureVideoQADataset(DATA_JSON, APPEARANCE_DIR, MOTION_DIR, tokenizer_name=MODEL_TEXT, max_len=MAX_LEN)
    
    n = len(full_ds)
    n_val = max(1, int(n * VALID_SPLIT))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    model = ImprovedVideoQA(text_model_name=MODEL_TEXT).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Optimizer with different learning rates for different parts
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
            "lr": LR,
        },
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": LR,
        },
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" not in n and not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
            "lr": LR * 2,
        },
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" not in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": LR * 2,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)
    
    # Cosine scheduler with warmup
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    
    # Loss function
    criterion = FocalLoss()
    scaler = GradScaler(enabled=USE_FP16)
    
    best_val = 0.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            if batch is None:
                continue
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast(enabled=USE_FP16):
                logits = model(input_ids, attention_mask, appearance, motion)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (step + 1), "lr": scheduler.get_last_lr()[0]})
            
            # Log to wandb
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        # Validation
        val_acc = evaluate_advanced(model, val_loader, device)
        print(f"Epoch {epoch+1} val_acc: {val_acc:.4f}")
        
        wandb.log({"val_accuracy": val_acc})
        
        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
            }, os.path.join(OUTPUT_DIR, "best_advanced.pt"))
            print("✅ Saved best advanced checkpoint.")
    
    print(f"🎉 Advanced training done. Best val: {best_val:.4f}")
    wandb.finish()

def evaluate_advanced(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                continue
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask, appearance, motion)
            preds = logits.argmax(dim=1)
            mask = labels >= 0
            
            if mask.sum().item() == 0:
                continue
                
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
    
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    train_advanced()