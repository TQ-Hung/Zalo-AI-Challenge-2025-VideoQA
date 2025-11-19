# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/train_self_training.py
import os
import math
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer
from datasets import collate_fn
from model import EarlyFusionQA
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json

# ---------- Combined Dataset Class ----------
class CombinedQADataset(Dataset):
    def __init__(self, original_json, pseudo_labels_json, appearance_dir, motion_dir, 
                 tokenizer_name="vinai/phobert-base", max_len=64):
        super().__init__()
        
        # Load original training data
        with open(original_json, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            if "data" in original_data:
                original_items = original_data["data"]
            else:
                original_items = original_data
        
        # Load pseudo-labels
        with open(pseudo_labels_json, 'r', encoding='utf-8') as f:
            pseudo_data = json.load(f)
            if "data" in pseudo_data:
                pseudo_items = pseudo_data["data"]
            else:
                pseudo_items = pseudo_data
        
        # Combine datasets - mark pseudo-labeled data
        self.items = []
        for item in original_items:
            item["is_pseudo"] = False
            self.items.append(item)
            
        for item in pseudo_items:
            item["is_pseudo"] = True
            # Convert answer_index to answer for compatibility
            if "answer_index" in item and "answer" not in item:
                item["answer"] = item["choices"][item["answer_index"]]
            self.items.append(item)
        
        self.appearance_dir = appearance_dir
        self.motion_dir = motion_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        
        print(f"✅ Combined dataset: {len(original_items)} original + {len(pseudo_items)} pseudo-labels = {len(self.items)} total")

    def __len__(self):
        return len(self.items)

    def _load_npy_safe(self, path):
        if not os.path.exists(path):
            return None
        try:
            arr = np.load(path)
            return arr if arr is not None and arr.size > 0 else None
        except Exception as e:
            print(f"Warning: Could not load {path}, error: {e}")
            return None

    def __getitem__(self, idx):
        it = self.items[idx]
        vid_basename = os.path.splitext(os.path.basename(it["video_path"]))[0]
        
        app_path = os.path.join(self.appearance_dir, f"{vid_basename}.npy")
        mot_path = os.path.join(self.motion_dir, f"{vid_basename}.npy")

        app_arr = self._load_npy_safe(app_path)
        mot_arr = self._load_npy_safe(mot_path)

        if app_arr is None:
            app_arr = np.zeros((1, 2048), dtype=np.float32)
        if mot_arr is None:
            mot_arr = np.zeros((1, 2048), dtype=np.float32)

        app_feat = torch.tensor(app_arr, dtype=torch.float32)
        mot_feat = torch.tensor(mot_arr, dtype=torch.float32)   

        question = it["question"]
        choices = it["choices"]
        texts = [question + " " + c for c in choices]
        enc = self.tokenizer(texts, padding="max_length", truncation=True,
                            max_length=self.max_len, return_tensors="pt")

        label = -1
        ans = it.get("answer", None)
        if ans is not None:
            for i, c in enumerate(choices):
                if str(ans).strip() == str(c).strip():
                    label = i
                    break
            # If still not found and we have answer_index, use that
            if label == -1 and "answer_index" in it:
                label = it["answer_index"]

        return {
            "video_id": it.get("id", vid_basename),
            "appearance": app_feat,
            "motion": mot_feat,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": label,
            "choices": choices,
            "is_pseudo": it.get("is_pseudo", False),
            "confidence": it.get("confidence", 1.0)
        }

# ---------- Config ----------
MODEL_TEXT = "vinai/phobert-base"
BATCH_SIZE = 8
MAX_LEN = 64
LR = 2e-5
EPOCHS = 20
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "/kaggle/working/checkpoints_self_training"
VALID_SPLIT = 0.1
SEED = 42

USE_FP16 = True
ACCUM_STEPS = 2
UNFREEZE_LAST_N = 3
EARLYSTOP_PATIENCE = 3
CLIP_NORM = 1.0
NUM_WORKERS = 4

# Self-training specific config
PSEUDO_LABELS_JSON = "/kaggle/working/pseudo_labels.json"
ORIGINAL_TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
APPEARANCE_DIR = "/kaggle/working/features_v2/appearance"
MOTION_DIR = "/kaggle/working/features_v2/motion"
PSEUDO_WEIGHT = 0.5

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def unfreeze_last_n(text_encoder, n=3):
    for param in text_encoder.parameters():
        param.requires_grad = False
    try:
        for layer in text_encoder.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(text_encoder, "pooler"):
            for p in text_encoder.pooler.parameters():
                p.requires_grad = True
    except Exception:
        params = list(text_encoder.parameters())
        for p in params[-n:]:
            p.requires_grad = True

def weighted_loss(logits, labels, is_pseudo, pseudo_weight=0.5):
    base_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    
    # Apply weights: pseudo-labeled samples get lower weight
    weights = torch.ones_like(base_loss)
    if is_pseudo is not None:
        mask = torch.tensor(is_pseudo, device=logits.device, dtype=torch.bool)
        weights[mask] = pseudo_weight
    
    return (base_loss * weights).mean()

def evaluate_self_training(model, loader, device):
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

def train_with_pseudo_labels():
    seed_everything()
    
    print("📚 Loading combined dataset (original + pseudo-labels)...")
    combined_ds = CombinedQADataset(
        original_json=ORIGINAL_TRAIN_JSON,
        pseudo_labels_json=PSEUDO_LABELS_JSON,
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
    
    train_ds = Subset(combined_ds, train_idx)
    val_ds = Subset(combined_ds, val_idx)
    
    print(f"📊 Dataset split: {len(train_ds)} train, {len(val_ds)} validation")
    
    # Count pseudo-labels in train set
    pseudo_count = sum(1 for i in train_idx if combined_ds.items[i].get("is_pseudo", False))
    print(f"📝 Pseudo-labels in training set: {pseudo_count}/{len(train_ds)} ({pseudo_count/len(train_ds)*100:.1f}%)")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"🔧 Device: {device}, GPUs available: {ngpu}")
    
    # Build model
    model = EarlyFusionQA(text_model_name=MODEL_TEXT)
    
    # Unfreeze last n layers
    if hasattr(model, "text_encoder"):
        unfreeze_last_n(model.text_encoder, UNFREEZE_LAST_N)
        print(f"🔓 Unfroze last {UNFREEZE_LAST_N} layers of text encoder.")
    model.to(device)
    
    # Wrap with DataParallel if multiple GPUs
    if ngpu > 1:
        model = torch.nn.DataParallel(model)
        print(f"🔄 Using DataParallel with {ngpu} GPUs")
    
    # Prepare optimizer parameter groups
    model_for_params = model.module if hasattr(model, "module") else model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model_for_params.text_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
            "lr": LR,
        },
        {
            "params": [
                p for n, p in model_for_params.text_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": LR,
        },
        {
            "params": [
                p for n, p in model_for_params.named_parameters()
                if not n.startswith("text_encoder")
            ],
            "weight_decay": WEIGHT_DECAY,
            "lr": LR * 5,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
    scaler = GradScaler(enabled=USE_FP16)
    
    best_val = 0.0
    epochs_no_improve = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Starting training with pseudo-labels...")
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Self-Training]")
        total_loss = 0.0
        optimizer.zero_grad()
        
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
                loss = weighted_loss(logits, labels, is_pseudo, PSEUDO_WEIGHT)
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({
                "loss": total_loss / (step + 1),
                "pseudo_weight": PSEUDO_WEIGHT
            })
        
        # Validation
        val_acc = evaluate_self_training(model, val_loader, device)
        print(f"📊 Epoch {epoch+1} val_acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        # Early stopping + save best
        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            save_path = os.path.join(OUTPUT_DIR, "best_self_trained.pt")
            to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({
                "model": to_save,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "pseudo_weight": PSEUDO_WEIGHT
            }, save_path)
            print("✅ Saved best self-trained checkpoint.")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= EARLYSTOP_PATIENCE:
                print("🛑 Early stopping triggered. Stopping training.")
                break
    
    print(f"🎉 Self-training done. Best val: {best_val:.4f}")

if __name__ == "__main__":
    train_with_pseudo_labels()