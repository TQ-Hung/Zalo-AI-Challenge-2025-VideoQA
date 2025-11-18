import os, random, torch, numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn
from model import CrossModalQA
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# ---------- Config ----------
APPEARANCE_DIR = "/kaggle/working/features_v2/appearance"
MOTION_DIR = "/kaggle/working/features_v2/motion"
MODEL_TEXT = "vinai/phobert-base-v2"

BATCH_SIZE = 16
MAX_LEN = 64
LR = 2e-5
EPOCHS = 30
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "checkpoints"
VALID_SPLIT = 0.1
SEED = 42
USE_FP16 = True
ACCUM_STEPS = 2
EARLYSTOP_PATIENCE = 5
CLIP_NORM = 1.0
NUM_WORKERS = 4


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Training ----------
def train():
    seed_everything()

    # set clean_up_tokenization_spaces to remove the FutureWarning
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT, trust_remote_code=True, clean_up_tokenization_spaces=True)

    full_ds = FeatureVideoQADataset(
        "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json",
        APPEARANCE_DIR,
        MOTION_DIR,
        tokenizer_name=MODEL_TEXT,
        max_len=MAX_LEN,
    )

    train_idx, val_idx = train_test_split(range(len(full_ds)), test_size=VALID_SPLIT, random_state=SEED)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {ngpu}")

    model = CrossModalQA(text_encoder_name=MODEL_TEXT).to(device)

    if ngpu > 1:
        model = torch.nn.DataParallel(model)

    model_for_params = model.module if hasattr(model, "module") else model

    optimizer = torch.optim.AdamW(model_for_params.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, verbose=True
    )

    scaler = GradScaler()  # device set automatically; use default

    best_val = 0.0
    epochs_no_improve = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # batch contains: input_ids (B, C, L), attention_mask (B, C, L), appearance_feats (B, T, 768), motion_feats (B, T, 768), labels (B)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            labels = batch["labels"].to(device)

            # build video_feats as (B, T, 768) by concatenating along feature dim if needed
            # if your dataset stores appearance and motion as separate temporal streams, you may instead fuse them.
            # here we concatenate the feature-dim (last dim) if they have same time length, else we mean-pool then concat.
            if appearance.dim() == 3 and motion.dim() == 3 and appearance.size(1) == motion.size(1):
                # simple average fusion across streams
                video_feats = (appearance + motion) / 2.0  # (B, T, 768)
            else:
                # fallback: mean pool temporally then concatenate
                a_pool = appearance.mean(dim=1)
                m_pool = motion.mean(dim=1)
                # create a single time step
                video_feats = torch.stack([a_pool, m_pool], dim=1)  # (B, 2, 768)

            # Flatten text inputs for encoder if needed inside model as well; model can handle both, but to be explicit we keep (B, C, L)

            with autocast(enabled=USE_FP16):
                logits = model(video_feats, input_ids, attention_mask)  # (B, C)
                loss = F.cross_entropy(logits, labels)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"loss": total_loss / (step + 1)})

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} val_acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            save_path = os.path.join(OUTPUT_DIR, "best.pt")
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
                print("Early stopping triggered.")
                break

    print("Training done. Best val:", best_val)


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            labels = batch["labels"].to(device)

            if appearance.dim() == 3 and motion.dim() == 3 and appearance.size(1) == motion.size(1):
                video_feats = (appearance + motion) / 2.0
            else:
                a_pool = appearance.mean(dim=1)
                m_pool = motion.mean(dim=1)
                video_feats = torch.stack([a_pool, m_pool], dim=1)

            logits = model(video_feats, input_ids, attention_mask)
            preds = logits.argmax(dim=1)

            mask = labels >= 0
            if mask.sum() == 0:
                continue

            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    train()
