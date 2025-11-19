# src/train.py
import os
import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn
from model import EarlyFusionQA
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from constants import DATA_JSON

# ---------- Config ----------
APPEARANCE_DIR = "features_v2/appearance"
MOTION_DIR = "features_v2/motion"
# train.py
MODEL_TEXT = "vinai/phobert-base"
BATCH_SIZE = 8
MAX_LEN = 64
LR = 2e-5
EPOCHS = 20
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "checkpoints"
VALID_SPLIT = 0.1
SEED = 42

USE_FP16 = True         # enable AMP
ACCUM_STEPS = 2         # gradient accumulation
UNFREEZE_LAST_N = 3     # unfreeze last N layers of BERT
EARLYSTOP_PATIENCE = 3  # early stopping patience (epochs)
CLIP_NORM = 1.0         # gradient clipping
NUM_WORKERS = 4
# ----------------------------

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def unfreeze_last_n(text_encoder, n=3):
    # freeze all first
    for param in text_encoder.parameters():
        param.requires_grad = False
    # unfreeze last n layers
    try:
        # huggingface bert-like: encoder.layer is a ModuleList
        for layer in text_encoder.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        # also unfreeze pooler if exists
        if hasattr(text_encoder, "pooler"):
            for p in text_encoder.pooler.parameters():
                p.requires_grad = True
    except Exception:
        # fallback: unfreeze last n parameters if structure unknown
        params = list(text_encoder.parameters())
        for p in params[-n:]:
            p.requires_grad = True

def train():
    seed_everything()

    # tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT)
    full_ds = FeatureVideoQADataset(
        DATA_JSON, APPEARANCE_DIR, MOTION_DIR,
        tokenizer_name=MODEL_TEXT, max_len=MAX_LEN
    )

    n = len(full_ds)
    n_val = max(1, int(n * VALID_SPLIT))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    print("Train data loading ...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    print("Val data loading ...")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"Device: {device}, GPUs available: {ngpu}")

    # build model and optionally unfreeze some BERT layers
    model = EarlyFusionQA(text_model_name=MODEL_TEXT)

    # unfreeze last n layers on model.text_encoder BEFORE wrapping with DataParallel
    if hasattr(model, "text_encoder"):
        unfreeze_last_n(model.text_encoder, UNFREEZE_LAST_N)
        print(f"Unfroze last {UNFREEZE_LAST_N} layers of text encoder.")
    model.to(device)

    # wrap with DataParallel if multiple GPUs
    if ngpu > 1:
        model = torch.nn.DataParallel(model)
        print("Using DataParallel with", ngpu, "GPUs")

    # Prepare optimizer parameter groups:
    # handle DataParallel (module) vs single model
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

    # Use ReduceLROnPlateau to reduce LR when val acc plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    scaler = GradScaler(enabled=USE_FP16)

    best_val = 0.0
    epochs_no_improve = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train")
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=USE_FP16):
                logits = model(input_ids, attention_mask, appearance, motion)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss / ACCUM_STEPS

            # backward with scaler
            scaler.scale(loss).backward()

            # gradient accumulation step
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                # unscale before clipping
                scaler.unscale_(optimizer)
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"loss": total_loss / (step + 1)})

        # validation
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} val_acc: {val_acc:.4f}")

        # scheduler step based on val metric
        scheduler.step(val_acc)

        # early stopping + save best
        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            save_path = os.path.join(OUTPUT_DIR, "best.pt")
            # if DataParallel, save module's state_dict
            to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({
                "model": to_save,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
            }, save_path)
            print("✅ Saved best checkpoint.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= EARLYSTOP_PATIENCE:
                print("Early stopping triggered. Stopping training.")
                break

    print("Training done. Best val:", best_val)

def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
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
    train()
