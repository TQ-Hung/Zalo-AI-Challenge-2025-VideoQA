# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/train_self_training_improved.py
import os
import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import FeatureVideoQADataset, collate_fn
from model_improved import ImprovedVideoQA
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ---------- Config ----------
MODEL_TEXT = "vinai/phobert-base"
BATCH_SIZE = 16
MAX_LEN = 128
LR = 3e-5
EPOCHS = 20
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "/kaggle/working/checkpoints_self_training_improved"
VALID_SPLIT = 0.1
SEED = 42

USE_FP16 = True
ACCUM_STEPS = 1
WARMUP_STEPS = 100
CLIP_NORM = 1.0
NUM_WORKERS = 4

# Self-training specific config
PSEUDO_WEIGHT = 0.5  # Default, can be overridden

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def weighted_loss(logits, labels, is_pseudo, pseudo_weight=0.5):
    base_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    
    # Apply weights: pseudo-labeled samples get lower weight
    weights = torch.ones_like(base_loss)
    if is_pseudo is not None:
        mask = torch.tensor(is_pseudo, device=logits.device, dtype=torch.bool)
        weights[mask] = pseudo_weight
    
    return (base_loss * weights).mean()

def evaluate_self_training_improved(model, loader, device):
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

def train_with_pseudo_labels_improved(original_train_json, pseudo_labels_json, output_dir, pseudo_weight=0.5):
    seed_everything()
    
    # Use the same feature directories as in the original training
    APPEARANCE_DIR = "/kaggle/working/features_v2/appearance"
    MOTION_DIR = "/kaggle/working/features_v2/motion"
    
    print("📚 Loading combined dataset (original + pseudo-labels)...")
    from combined_dataset import CombinedQADataset
    combined_ds = CombinedQADataset(
        original_json=original_train_json,
        pseudo_labels_json=pseudo_labels_json,
        appearance_dir=APPEARANCE_DIR,
        motion_dir=MOTION_DIR,
        tokenizer_name=MODEL_TEXT,
        max_len=MAX_LEN
    )
    
    # Train-validation split
    n = len(combined_ds)
    n_val = max(1, int(n * VALID_SPLIT))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    from torch.utils.data import Subset
    train_ds = Subset(combined_ds, train_idx)
    val_ds = Subset(combined_ds, val_idx)
    
    print(f"📊 Dataset split: {len(train_ds)} train, {len(val_ds)} validation")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"🔧 Device: {device}, GPUs available: {ngpu}")
    
    # Build model
    model = ImprovedVideoQA(text_model_name=MODEL_TEXT).to(device)
    
    # Wrap with DataParallel if multiple GPUs
    if ngpu > 1:
        model = torch.nn.DataParallel(model)
        print(f"🔄 Using DataParallel with {ngpu} GPUs")
    
    # Optimizer
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
    
    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=USE_FP16)
    
    best_val = 0.0
    epochs_no_improve = 0
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting training with pseudo-labels (ImprovedVideoQA)...")
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Self-Training Improved]")
        total_loss = 0.0
        
        for step, batch in enumerate(pbar):
            if batch is None:
                continue
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            labels = batch["labels"].to(device)
            
            # Get pseudo-label flags
            is_pseudo = batch.get("is_pseudo", [False] * len(labels))
            
            with autocast(enabled=USE_FP16):
                logits = model(input_ids, attention_mask, appearance, motion)
                loss = weighted_loss(logits, labels, is_pseudo, pseudo_weight)
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": total_loss / (step + 1),
                "pseudo_weight": pseudo_weight,
                "lr": scheduler.get_last_lr()[0]
            })
        
        # Validation
        val_acc = evaluate_self_training_improved(model, val_loader, device)
        print(f"📊 Epoch {epoch+1} val_acc: {val_acc:.4f}")
        
        # Early stopping + save best
        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            save_path = os.path.join(output_dir, "best_self_trained.pt")
            to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({
                "model": to_save,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "pseudo_weight": pseudo_weight
            }, save_path)
            print("✅ Saved best self-trained checkpoint.")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= 3:  # early stopping patience
                print("🛑 Early stopping triggered. Stopping training.")
                break
    
    print(f"🎉 Self-training done. Best val: {best_val:.4f}")