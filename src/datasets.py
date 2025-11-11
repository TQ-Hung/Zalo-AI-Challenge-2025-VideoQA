# src/datasets.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

# --- helper: pad video features (time dim) ---
def pad_sequence_feats(feat_list):
    """
    feat_list: list of tensors (T_i, D)
    return: (B, T_max, D)
    """
    if len(feat_list) == 0:
        return torch.zeros((0, 0, 0), dtype=torch.float32)
    max_len = max(f.shape[0] for f in feat_list)
    feat_dim = feat_list[0].shape[1]
    batch_size = len(feat_list)
    padded = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    for i, f in enumerate(feat_list):
        L = f.shape[0]
        padded[i, :L] = f
    return padded

class FeatureVideoQADataset(Dataset):
    """
    Compatible with train.py usage:
    FeatureVideoQADataset(DATA_JSON, APPEARANCE_DIR, MOTION_DIR, tokenizer_name=MODEL_TEXT, max_len=MAX_LEN)
    Returns items that collate_fn expects for training (multi-choice).
    """
    def __init__(self, json_path, appearance_dir, motion_dir,
                 tokenizer_name="vinai/phobert-base-v2", max_len=64, is_test=False):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # accept both {"data": [...]} or a list directly
            if isinstance(data, dict) and "data" in data:
                self.items = data["data"]
            elif isinstance(data, list):
                self.items = data
            else:
                raise ValueError("Unexpected json format for dataset.")
        self.appearance_dir = appearance_dir
        self.motion_dir = motion_dir
        # adjust path if needed
        self.ocr_dir = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/features_v2/ocr"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.items)

    def _load_npy_safe(self, path):
        try:
            if not os.path.exists(path):
                return None
            arr = np.load(path)
            if arr is None:
                return None
            # ensure 2D
            if arr.ndim == 1:
                arr = np.expand_dims(arr, 0)
            return arr.astype(np.float32)
        except Exception:
            return None

    def __getitem__(self, idx):
        it = self.items[idx]
        question = it.get("question", "")
        vid_path = it.get("video_path", "")
        vid_basename = os.path.splitext(os.path.basename(vid_path))[0]

        app_path = os.path.join(self.appearance_dir, f"{vid_basename}.npy")
        mot_path = os.path.join(self.motion_dir, f"{vid_basename}.npy")

        app_arr = self._load_npy_safe(app_path)
        mot_arr = self._load_npy_safe(mot_path)

        # fallback to tiny zeros if missing
        if app_arr is None:
            app_arr = np.zeros((1, 768), dtype=np.float32)
        if mot_arr is None:
            mot_arr = np.zeros((1, 768), dtype=np.float32)

        app_feat = torch.tensor(app_arr, dtype=torch.float32)
        mot_feat = torch.tensor(mot_arr, dtype=torch.float32)

        # augmentation (train only)
        if not self.is_test and random.random() < 0.5:
            noise_level = 0.03
            app_feat = app_feat + torch.randn_like(app_feat) * noise_level
            mot_feat = mot_feat + torch.randn_like(mot_feat) * noise_level

        # OCR injection
        ocr_path = os.path.join(self.ocr_dir, f"{vid_basename}.txt")
        if os.path.exists(ocr_path):
            try:
                with open(ocr_path, "r", encoding="utf-8") as f:
                    ocr_text = f.read().strip()
                if ocr_text:
                    question = f"[OCR: {ocr_text}] {question}"
            except Exception:
                pass

        # choices: expect list of strings; if not present, fallback to single-text classification
        choices = it.get("choices", None)
        if choices is None:
            # fallback: if options stored as "options" or not stored, try common names
            choices = it.get("options", None)
        if choices is None:
            # no choices: convert to single-choice with 1 option (shouldn't be usual)
            choices = [question]

        # Build per-choice encoded inputs (C, L)
        texts = [f"{question} {c}".strip() for c in choices]
        enc = self.tokenizer(texts, padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")

        # label mapping: if answer given as text matching one choice, find index; if "A/B/C" style, also support
        label = -1
        ans = it.get("answer", None)
        if ans is not None:
            # if ans is index-like (int), use it
            if isinstance(ans, int):
                label = ans if 0 <= ans < len(choices) else -1
            else:
                ans_s = str(ans).strip()
                # support "A"/"B" -> index
                if ans_s.upper() in ["A", "B", "C", "D", "E"]:
                    label = ord(ans_s.upper()) - ord("A")
                    if label >= len(choices):
                        label = -1
                else:
                    # match by exact text
                    for i, c in enumerate(choices):
                        if ans_s == c or ans_s.strip() == c.strip():
                            label = i
                            break

        if self.is_test:
            return {
                "ids": it.get("id", vid_basename),
                "input_ids": enc["input_ids"],         # (C, L)
                "attention_mask": enc["attention_mask"],# (C, L)
                "appearance": app_feat,                 # (T_app, D)
                "motion": mot_feat,                     # (T_mot, D)
            }
        else:
            return {
                "video_id": it.get("id", vid_basename),
                "appearance": app_feat,
                "motion": mot_feat,
                "input_ids": enc["input_ids"],          # (C, L)
                "attention_mask": enc["attention_mask"],# (C, L)
                "label": label,
                "choices": choices,
            }

# --- collate_fn used by train.py ---
def collate_fn(batch):
    # If some elements are None, filter
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Detect training vs inference by presence of "choices"
    if "choices" not in batch[0]:
        # inference/test path: each input_ids is (C, L). We will stack into (B, C, L)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        appearance_feats = pad_sequence_feats([b["appearance"] for b in batch])
        motion_feats = pad_sequence_feats([b["motion"] for b in batch])
        ids = [b["ids"] for b in batch]
        return {
            "ids": ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "appearance_feats": appearance_feats,
            "motion_feats": motion_feats,
        }

    # train/val path: pad choices to same max_choices -> produce (B, C, L)
    max_choices = max(len(b["choices"]) for b in batch)
    padded_input_ids = []
    padded_attention_masks = []
    for b in batch:
        c, L = b["input_ids"].shape
        if c < max_choices:
            pad_c = max_choices - c
            pad_ids = torch.zeros((pad_c, L), dtype=torch.long)
            pad_mask = torch.zeros((pad_c, L), dtype=torch.long)
            ids_tensor = torch.cat([b["input_ids"], pad_ids], dim=0)
            mask_tensor = torch.cat([b["attention_mask"], pad_mask], dim=0)
        else:
            ids_tensor = b["input_ids"]
            mask_tensor = b["attention_mask"]
        padded_input_ids.append(ids_tensor)
        padded_attention_masks.append(mask_tensor)

    input_ids = torch.stack(padded_input_ids, dim=0)        # (B, C, L)
    attention_mask = torch.stack(padded_attention_masks, dim=0)  # (B, C, L)

    appearance_feats = pad_sequence_feats([b["appearance"] for b in batch])
    motion_feats = pad_sequence_feats([b["motion"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,               # (B, C, L)
        "attention_mask": attention_mask,     # (B, C, L)
        "appearance_feats": appearance_feats, # (B, T_app, D)
        "motion_feats": motion_feats,         # (B, T_mot, D)
        "labels": labels,
    }

def collate_fn_inference(batch):
    # Bỏ các sample lỗi
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Flatten tất cả câu hỏi trong batch
    ids = []
    input_ids, attention_masks = [], []
    appearance_feats, motion_feats = [], []

    for b in batch:
        num_q = b["input_ids"].shape[0]
        ids.extend(b["ids"])
        input_ids.append(b["input_ids"])
        attention_masks.append(b["attention_mask"])

        # Lặp lại feature cho mỗi câu hỏi
        app_feat = b["appearance"].unsqueeze(0).repeat(num_q, 1, 1)
        mot_feat = b["motion"].unsqueeze(0).repeat(num_q, 1, 1)

        appearance_feats.append(app_feat)
        motion_feats.append(mot_feat)

    # Nối tất cả lại thành batch phẳng
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)
    appearance_feats = torch.cat(appearance_feats, dim=0)
    motion_feats = torch.cat(motion_feats, dim=0)

    return {
        "ids": ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "appearance_feats": appearance_feats,
        "motion_feats": motion_feats
    }