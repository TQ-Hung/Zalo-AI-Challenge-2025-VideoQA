# src/dataset_features.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def pad_sequence_feats(feat_list):
    """
    feat_list: list of tensors có shape (T_i, D)
    Mục tiêu: pad tất cả lên T_max theo batch để có (B, T_max, D)
    Đồng thời đảm bảo tất cả features có cùng dimension D = 768.
    """
    # Kiểm tra dimension của các features, nếu có feature nào không phải 768 thì chuyển về 768
    for i, f in enumerate(feat_list):
        if f.shape[1] != 768:
            # Nếu dimension lớn hơn 768, cắt bớt
            if f.shape[1] > 768:
                feat_list[i] = f[:, :768]
            # Nếu dimension nhỏ hơn 768, pad thêm bằng 0
            else:
                pad_size = 768 - f.shape[1]
                feat_list[i] = torch.nn.functional.pad(f, (0, pad_size), "constant", 0)

    max_len = max(f.shape[0] for f in feat_list)
    feat_dim = 768  # Giờ đây tất cả đều là 768
    batch_size = len(feat_list)

    padded = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    for i, f in enumerate(feat_list):
        length = f.shape[0]
        padded[i, :length] = f
    return padded

class FeatureVideoQADataset(Dataset):
    """
    Loads:
      - appearance .npy -> shape (T_app, 2048)
      - motion .npy     -> shape (T_mot, 2048)
    Returns per sample:
      - app_feat_mean: torch.FloatTensor (2048,)
      - mot_feat_mean: torch.FloatTensor (2048,)
      - input_ids: torch.LongTensor (num_choices, L)
      - attention_mask: torch.LongTensor (num_choices, L)
      - label: int (0..num_choices-1) or -1 if not provided
    """
    def __init__(self, json_path, appearance_dir, motion_dir,
                 tokenizer_name="bert-base-multilingual-cased", max_len=64, is_test=False):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)["data"]
        self.appearance_dir = appearance_dir
        self.motion_dir = motion_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.items)

    def _load_npy_safe(self, path):
        if not os.path.exists(path):
            return None
        arr = np.load(path)
        if arr is None or arr.size == 0:
            return None
        return arr

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
                if ans.strip() == c.strip():
                    label = i
                    break
        if self.is_test:
            return {
                "ids": it["id"],
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "appearance": app_feat,
                "motion": mot_feat,
            }
        else:
            return {
                "video_id": it.get("id", vid_basename),
                "appearance": app_feat,
                "motion": mot_feat,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": label,
                "choices": choices,
            }


import torch
import torch.nn.functional as F

# --- Hàm pad chuỗi feature theo chiều thời gian ---
def pad_sequence_feats(feat_list):
    """
    feat_list: list of tensors có shape (T_i, D)
    Mục tiêu: pad tất cả lên T_max theo batch để có (B, T_max, D)
    """
    max_len = max(f.shape[0] for f in feat_list)
    feat_dim = feat_list[0].shape[1]
    batch_size = len(feat_list)

    padded = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    for i, f in enumerate(feat_list):
        length = f.shape[0]
        padded[i, :length] = f
    return padded


# --- Collate function cho train/val ---
def collate_fn(batch):
    # Nếu batch của test set (không có 'choices') thì xử lý riêng
    if "choices" not in batch[0]:
        # Inference/test mode
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_masks = torch.stack([b["attention_mask"] for b in batch])
        appearance_feats = pad_sequence_feats([b["appearance"] for b in batch])
        motion_feats = pad_sequence_feats([b["motion"] for b in batch])
        ids = [b["ids"] for b in batch]
        return {
            "ids": ids,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "appearance_feats": appearance_feats,
            "motion_feats": motion_feats,
        }

    # --- Training/validation mode ---
    batch = [b for b in batch if "choices" in b and len(b["choices"]) > 0]
    if len(batch) == 0:
        return None

    max_choices = max(len(b["choices"]) for b in batch)
    input_ids, attention_masks = [], []

    for b in batch:
        pad_len = max_choices - len(b["choices"])
        if pad_len > 0:
            pad_ids = torch.zeros((pad_len, b["input_ids"].shape[1]), dtype=torch.long)
            pad_mask = torch.zeros((pad_len, b["attention_mask"].shape[1]), dtype=torch.long)
            input_ids.append(torch.cat([b["input_ids"], pad_ids], dim=0))
            attention_masks.append(torch.cat([b["attention_mask"], pad_mask], dim=0))
        else:
            input_ids.append(b["input_ids"])
            attention_masks.append(b["attention_mask"])

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    # ✅ Pad theo chiều thời gian cho video features
    appearance_feats = pad_sequence_feats([b["appearance"] for b in batch])
    motion_feats = pad_sequence_feats([b["motion"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,            # (B, C, L)
        "attention_mask": attention_masks, # (B, C, L)
        "appearance_feats": appearance_feats,  # (B, T_app_max, 2048)
        "motion_feats": motion_feats,          # (B, T_mot_max, 2048)
        "labels": labels,
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
