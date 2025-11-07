# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel
import os

# --- Temporal Aggregation Module ---
class TemporalAggregator(nn.Module):
    """
    Áp dụng cho chuỗi frame (appearance hoặc motion) để gộp thông tin theo thời gian.
    Gồm: Linear giảm chiều → BiGRU → Attention pooling
    """
    def __init__(self, in_dim=768, hidden_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, D)
        x = self.proj(x)
        out, _ = self.gru(x)
        attn_score = self.attn(out)  # (B, T, 1)
        attn_weight = torch.softmax(attn_score, dim=1)
        pooled = torch.sum(out * attn_weight, dim=1)
        return pooled  # (B, hidden_dim*2)


# --- Main Early Fusion QA Model ---
class EarlyFusionQA(nn.Module):
    """
    EarlyFusionQA + Temporal Aggregation (BiGRU + Attention)
    - Text: Transformer encoder (PhoBERT, etc.)
    - Video: DINOv2 (appearance) + VideoMAE (motion)
    - Fusion: Gated residual + Transformer fusion
    """
    def __init__(self,
                 text_model_name="bert-base-multilingual-cased",
                 video_dim=768,         # ✅ đổi từ 2048 -> 768
                 text_dim=768,
                 hidden_dim=512,
                 fusion_dim=512,
                 dropout=0.2,
                 text_pooling="cls"):
        super().__init__()
        # --- Text Encoder ---
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN", None),
            trust_remote_code=True
        )

        self.text_pooling = text_pooling

        # --- Temporal aggregation for video ---
        self.temporal_agg = TemporalAggregator(in_dim=video_dim, hidden_dim=hidden_dim // 2)
        self.video_dropout = nn.Dropout(dropout)

        # --- Text projection ---
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Fusion module ---
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )

    def pool_text(self, out, mask):
        """Lấy embedding cho câu hỏi + đáp án"""
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if self.text_pooling == "cls":
            return out.last_hidden_state[:, 0]
        elif self.text_pooling == "mean":
            mask = mask.unsqueeze(-1)
            return (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        else:
            raise ValueError("Unknown text pooling type")

    def forward(self, input_ids, attention_mask, appearance, motion):
        """
        appearance: (B, T_app, 768)
        motion: (B, T_mot, 768)
        """
        B, C, L = input_ids.shape
        device = input_ids.device

        # --- Text Encoding ---
        flat_input_ids = input_ids.view(B * C, L)
        flat_attention_mask = attention_mask.view(B * C, L)
        text_out = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled = self.pool_text(text_out, flat_attention_mask)
        text_feat = self.text_proj(pooled)  # (B*C, hidden_dim)

        # --- Video Temporal Encoding ---
        # Pad 2 chuỗi cho cùng độ dài
        T = max(appearance.size(1), motion.size(1))
        if appearance.size(1) < T:
            pad = torch.zeros(B, T - appearance.size(1), appearance.size(2), device=device)
            appearance = torch.cat([appearance, pad], dim=1)
        if motion.size(1) < T:
            pad = torch.zeros(B, T - motion.size(1), motion.size(2), device=device)
            motion = torch.cat([motion, pad], dim=1)

        # ✅ Trung bình appearance & motion
        video_seq = (appearance + motion) / 2
        video_feat = self.temporal_agg(video_seq)  # (B, hidden_dim)
        video_feat = self.video_dropout(video_feat)
        video_feat_rep = video_feat.unsqueeze(1).repeat(1, C, 1).view(B * C, -1)

        # --- Fusion ---
        fused_input = torch.cat([text_feat, video_feat_rep], dim=1)
        gate_out = torch.sigmoid(self.gate(fused_input))
        fused = gate_out * text_feat + (1 - gate_out) * video_feat_rep
        fused = self.fusion_norm(fused)

        fused = fused.view(B, C, -1)
        fused = self.fusion_transformer(fused)
        fused = fused.view(B * C, -1)

        logits = self.classifier(fused).view(B, C)
        return logits
