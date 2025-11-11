# src/inference.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import FeatureVideoQADataset, collate_fn_inference
from model import CrossModalQA as EarlyFusionQA

# ----- Config -----
MODEL_TEXT = "vinai/phobert-base-v2"
APPEARANCE_DIR = "/kaggle/working/features/appearance"
MOTION_DIR = "/kaggle/working/features/motion"
CHECKPOINT = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
BATCH_SIZE = 8
OUTPUT_FILE = "/kaggle/working/submission.json"
TTA_TIMES = 3  # số lần Test Time Augmentation
# -------------------

def inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from {CHECKPOINT} ...")
    model = EarlyFusionQA(text_model_name=MODEL_TEXT).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print("Preparing test dataset ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_TEXT,
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN", None),
        trust_remote_code=True,
        ignore_chat_template_errors=True
    )

    test_ds = FeatureVideoQADataset(
        TEST_JSON, APPEARANCE_DIR, MOTION_DIR,
        tokenizer_name=MODEL_TEXT, max_len=64, is_test=True
    )
    # Reuse collate_fn cho inference vì nó handle cả is_test
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_inference,
        num_workers=2
    )

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference + TTA"):
            if batch is None:
                continue
            ids = batch["ids"]
            all_logits = []

            for tta_id in range(TTA_TIMES):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # thêm noise nhẹ vào features cho TTA
                noise_level = 0.03 if tta_id > 0 else 0.0
                appearance_feats = batch["appearance_feats"].to(device)
                motion_feats = batch["motion_feats"].to(device)
                if noise_level > 0:
                    appearance_feats = appearance_feats + torch.randn_like(appearance_feats) * noise_level
                    motion_feats = motion_feats + torch.randn_like(motion_feats) * noise_level

                logits = model(input_ids, attention_mask, appearance_feats, motion_feats)
                all_logits.append(logits)

            # trung bình logits giữa các lần TTA
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            preds = avg_logits.argmax(dim=1).cpu().tolist()

            for qid, pred in zip(ids, preds):
                results.append({"id": qid, "answer": int(pred)})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Saved predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    inference()
