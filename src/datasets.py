# src/datasets.py
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

class FeatureVideoQADataset(Dataset):
    def __init__(self, json_path, appearance_dir, motion_dir, tokenizer_name="vinai/phobert-base-v2", max_len=64):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
        
        self.items = []
        for item in data:
            video_path = item["video_path"]
            basename = os.path.splitext(os.path.basename(video_path))[0]
            app_path = os.path.join(appearance_dir, f"{basename}.npy")
            mot_path = os.path.join(motion_dir, f"{basename}.npy")
            
            if not (os.path.exists(app_path) and os.path.exists(mot_path)):
                continue
                
            self.items.append({
                "question": item["question"],
                "options": item.get("options", ["A", "B", "C", "D"]),
                "answer": item.get("answer", "A"),
                "appearance_path": app_path,
                "motion_path": mot_path,
                "video_id": basename
            })
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN", None),
            trust_remote_code=True
        )
        self.max_len = max_len

        # THÊM DÒNG NÀY ĐỂ CÓ OCR (nếu có)
        self.ocr_dir = "features_v2/ocr"

    def __len__(self):
        return len(self.items)  # QUAN TRỌNG NHẤT – FIX LỖI LEN()

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load features
        appearance = np.load(item["appearance_path"]).astype(np.float32)
        motion = np.load(item["motion_path"]).astype(np.float32)
        
        # OCR (nếu có)
        ocr_text = ""
        ocr_path = os.path.join(self.ocr_dir, f"{item['video_id']}.txt")
        if os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_text = f.read().strip()
        
        question = item["question"]
        if ocr_text:
            question = f"[OCR: {ocr_text}] {question}"
        
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            question,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Label mapping
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        label = label_map.get(item["answer"], -1)
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "appearance_feats": torch.from_numpy(appearance),
            "motion_feats": torch.from_numpy(motion),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    appearance = torch.stack([b["appearance_feats"] for b in batch])
    motion = torch.stack([b["motion_feats"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "appearance_feats": appearance,
        "motion_feats": motion,
        "labels": labels
    }

# --- Collate function cho inference ---
def collate_fn_inference(batch):
    batch = [b for b in batch if "input_ids" in b]

    ids = [b["ids"] for b in batch]
    input_ids = [b["input_ids"] for b in batch]          # (num_choices, seq_len)
    attention_mask = [b["attention_mask"] for b in batch]
    appearance = [b["appearance"] for b in batch]        # (T_app, 2048)
    motion = [b["motion"] for b in batch]                # (T_mot, 2048)

    # Lấy kích thước lớn nhất trong batch
    max_choices = max(x.shape[0] for x in input_ids)
    max_len = max(x.shape[1] for x in input_ids)

    padded_input_ids, padded_attention_masks = [], []
    for ids_tensor, mask_tensor in zip(input_ids, attention_mask):
        num_choices, seq_len = ids_tensor.shape
        pad_choices = max_choices - num_choices
        pad_len = max_len - seq_len

        # Padding chiều seq_len
        if pad_len > 0:
            pad_seq = torch.zeros((num_choices, pad_len), dtype=torch.long)
            ids_tensor = torch.cat([ids_tensor, pad_seq], dim=1)
            mask_tensor = torch.cat([mask_tensor, pad_seq], dim=1)

        # Padding chiều num_choices
        if pad_choices > 0:
            pad_ids = torch.zeros((pad_choices, max_len), dtype=torch.long)
            ids_tensor = torch.cat([ids_tensor, pad_ids], dim=0)
            mask_tensor = torch.cat([mask_tensor, pad_ids], dim=0)

        padded_input_ids.append(ids_tensor)
        padded_attention_masks.append(mask_tensor)

    input_ids = torch.stack(padded_input_ids)
    attention_mask = torch.stack(padded_attention_masks)

    # ✅ Pad theo thời gian cho video features
    appearance = pad_sequence_feats(appearance)
    motion = pad_sequence_feats(motion)

    return {
        "ids": ids,
        "input_ids": input_ids,           # (B, C, L)
        "attention_mask": attention_mask, # (B, C, L)
        "appearance": appearance,         # (B, T_app_max, 2048)
        "motion": motion,                 # (B, T_mot_max, 2048)
    }
