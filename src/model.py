# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class CrossModalQA(nn.Module):
    def __init__(self, 
                 text_model_name="vinai/phobert-base-v2",
                 video_feat_dim=768, text_dim=768, hidden_dim=512,
                 n_heads=8, n_layers=3, dropout=0.2, text_pooling="cls"):
        super().__init__()

        # ----- Text encoder -----
        self.text_encoder = AutoModel.from_pretrained(text_model_name, return_dict=True)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # ----- Video projection -----
        self.video_proj = nn.Linear(video_feat_dim, hidden_dim)

        # ----- Cross-modal transformer -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.cross_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ----- Classifier -----
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)   # output logits per choice
        )

    # Trong file model.py → SỬA HÀM forward
    def forward(self, input_ids, attention_mask, appearance, motion, ocr_feat=None, face_feat=None):
        # DÙNG POOLER NHƯ KHI TRAIN
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output  # ← PHẢI DÙNG CÁI NÀY!

        # Hoặc nếu bạn không muốn dùng pooler → train lại với CLS
        # text_feat = text_out.last_hidden_state[:, 0, :]

        app_proj = self.video_proj(appearance)
        mot_proj = self.video_proj(motion)

        fused = text_feat + app_proj + mot_proj
        if ocr_feat is not None and face_feat is not None:
            ocr_proj = self.ocr_proj(ocr_feat)
            face_proj = self.face_proj(face_feat)
            fused = fused + ocr_proj + face_proj

        logits = self.classifier(fused)
        return logits
