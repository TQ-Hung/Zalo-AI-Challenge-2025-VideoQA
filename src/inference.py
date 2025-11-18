# src/inference.py (fixed & robust)
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from model import CrossModalQA

# ------------------- CONFIG -------------------
MODEL_TEXT = "vinai/phobert-base-v2"
CHECKPOINT = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
FEATURE_DIR = "/kaggle/working/features_test"
APP_DIR = f"{FEATURE_DIR}/appearance"
MOT_DIR = f"{FEATURE_DIR}/motion"
OCR_TEXT_DIR = f"{FEATURE_DIR}/ocr"

BATCH_SIZE = 32
TTA_TIMES = 3
OUTPUT_FILE = "/kaggle/working/submission.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 64

# ------------------- DATASET -------------------
class PublicTestDataset(Dataset):
    def __init__(self):
        with open(TEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
        self.items = []
        for item in data:
            video_id = os.path.splitext(os.path.basename(item["video_path"]))[0]
            app_path = os.path.join(APP_DIR, f"{video_id}.npy")
            mot_path = os.path.join(MOT_DIR, f"{video_id}.npy")
            if not (os.path.exists(app_path) and os.path.exists(mot_path)):
                continue
            ocr_text_path = os.path.join(OCR_TEXT_DIR, f"{video_id}.txt")
            self.items.append({
                "question_id": item["id"],
                "question": item["question"],
                "video_id": video_id,
                "app_path": app_path,
                "mot_path": mot_path,
                "ocr_text_path": ocr_text_path if os.path.exists(ocr_text_path) else None
            })
        print(f"LOADED {len(self.items)} TEST SAMPLES")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        appearance = np.load(item["app_path"]).astype(np.float32)
        motion = np.load(item["mot_path"]).astype(np.float32)
        ocr_text = ""
        if item["ocr_text_path"]:
            with open(item["ocr_text_path"], "r", encoding="utf-8") as f:
                ocr_text = f.read().strip()
        question = f"[OCR: {ocr_text}] {item['question']}" if ocr_text else item['question']
        return {
            "question_id": item["question_id"],
            "question": question,
            "appearance": torch.from_numpy(appearance),
            "motion": torch.from_numpy(motion),
        }

def collate_fn(batch):
    return {
        "qids": [b["question_id"] for b in batch],
        "questions": [b["question"] for b in batch],
        "appearance": torch.stack([b["appearance"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch]),
    }

# ------------------- HELPERS -------------------
def load_checkpoint_strict(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    # support different key names
    if isinstance(ck, dict):
        if "model" in ck:
            state = ck["model"]
        elif "state_dict" in ck:
            state = ck["state_dict"]
        else:
            # maybe it's already a state_dict
            state = ck
    else:
        state = ck
    # if saved from DataParallel, keys may have "module."
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # try stripping module.
        new_state = {}
        for k, v in state.items():
            k2 = k.replace("module.", "") if k.startswith("module.") else k
            new_state[k2] = v
        model.load_state_dict(new_state)
    return model

# ------------------- MAIN -------------------
def main():
    print(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT, trust_remote_code=True, clean_up_tokenization_spaces=True)

    test_ds = PublicTestDataset()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # build model and load checkpoint
    model = CrossModalQA(text_encoder_name=MODEL_TEXT).to(DEVICE)

    # robust checkpoint load
    model = load_checkpoint_strict(model, CHECKPOINT, DEVICE)
    model.eval()

    # if multiple GPUs available and you prefer DataParallel for inference uncomment:
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    all_preds = []
    # Debug counters
    printed_shapes = False

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            questions = batch["questions"]
            # tokenize (this returns tensors with shape [B, L])
            encoded = tokenizer(questions, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            input_ids = encoded["input_ids"]  # (B, L)
            attention_mask = encoded["attention_mask"]  # (B, L)

            # prepare video feats: fuse appearance & motion same as training
            appearance = batch["appearance"]  # (B, T, D) or (B, D) depending on feature extraction
            motion = batch["motion"]
            # ensure 3D: (B, T, D). If features are (B, D) expand to (B, 1, D)
            if appearance.dim() == 2:
                appearance = appearance.unsqueeze(1)
            if motion.dim() == 2:
                motion = motion.unsqueeze(1)

            # if time dims match, fuse by mean; else fallback pool
            if appearance.size(1) == motion.size(1):
                video_feats = (appearance + motion) / 2.0  # (B, T, D)
            else:
                a_pool = appearance.mean(dim=1)
                m_pool = motion.mean(dim=1)
                # stack as 1 or 2 time steps (B, T', D)
                video_feats = torch.stack([a_pool, m_pool], dim=1)

            # send to device
            video_feats = video_feats.to(DEVICE)

            # TTA (test time augment) — add tiny noise to features
            tta_logits = []
            for tta_idx in range(TTA_TIMES):
                # prepare input ids & attention masks for this TTA
                # if model expects per-choice inputs (B, C, L) we need to expand tokenized inputs
                # get num_choices from model if available, else assume 4
                num_choices = getattr(model, "num_choices", 4)
                # if input_ids is 2D but model expects 3D, expand to (B, C, L) by repeating the same question
                if input_ids.dim() == 2:
                    if num_choices > 1:
                        # expand along choices
                        B, L = input_ids.shape
                        input_ids_exp = input_ids.unsqueeze(1).expand(B, num_choices, L).contiguous()
                        attention_exp = attention_mask.unsqueeze(1).expand(B, num_choices, L).contiguous()
                    else:
                        input_ids_exp = input_ids
                        attention_exp = attention_mask
                elif input_ids.dim() == 3:
                    input_ids_exp = input_ids
                    attention_exp = attention_mask
                else:
                    raise ValueError(f"Unexpected input_ids dim: {input_ids.dim()}")

                # optionally add feature noise
                noise_level = 0.02 if tta_idx > 0 else 0.0
                if noise_level > 0:
                    noise = torch.randn_like(video_feats) * noise_level
                    video_t = (video_feats + noise).to(DEVICE)
                else:
                    video_t = video_feats

                # move inputs to device (flattening handled inside model)
                input_ids_t = input_ids_exp.to(DEVICE)
                attention_t = attention_exp.to(DEVICE)

                # forward (model expects (video_feats, input_ids, attention_mask))
                # model handles either (B, C, L) or flattened (B*C, L)
                logits = model(video_t, input_ids_t, attention_t)  # expected shape (B, C)
                # safety: ensure logits is 2D (B, C)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                if logits.dim() != 2:
                    # try to convert: if (B, C, 1) squeeze last dim
                    if logits.dim() == 3 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)
                    else:
                        raise ValueError(f"Unexpected logits shape: {logits.shape}")

                tta_logits.append(logits.cpu())

            # average TTA logits
            avg_logits = torch.stack(tta_logits, dim=0).mean(dim=0)  # (B, C) on CPU

            # debug print once
            if not printed_shapes:
                print("DEBUG shapes: input_ids", input_ids.shape, "video_feats", video_feats.shape, "avg_logits", avg_logits.shape)
                # print small sample distribution
                sample = avg_logits[: min(5, avg_logits.size(0))]
                print("DEBUG sample logits (first rows):")
                print(sample.numpy())
                printed_shapes = True

            preds = avg_logits.argmax(dim=1).numpy()  # (B,)
            all_preds.extend(preds.tolist())

    # map to labels and save
    id_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    final_labels = [id_to_label.get(int(p), "A") for p in all_preds]

    import pandas as pd
    qids = [item["question_id"] for item in test_ds.items]
    assert len(qids) == len(final_labels), f"len(qids)={len(qids)} vs len(labels)={len(final_labels)}"

    df = pd.DataFrame([{"question_id": qid, "answer": label}
                       for qid, label in zip(qids, final_labels)])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUBMISSION READY: {OUTPUT_FILE}")
    print(f"Distribution: A:{final_labels.count('A')}, B:{final_labels.count('B')}, C:{final_labels.count('C')}, D:{final_labels.count('D')}")

if __name__ == "__main__":
    main()
