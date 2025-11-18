import torch
import torch.nn as nn
from transformers import AutoModel


# ------------------------------------------------------------
# 1. Video Encoder: dùng 2 stream appearance + motion
# ------------------------------------------------------------
class VideoEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()

        # Linear projection to unify feature sizes
        self.proj = nn.Linear(input_dim, hidden_dim)

        # Use GRU to aggregate temporal information
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # LayerNorm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, video_feats):
        """
        video_feats: (B, T, 768)
        """
        x = self.proj(video_feats)
        x, _ = self.gru(x)
        x = self.norm(x)
        return x  # (B, T, hidden_dim)


# ------------------------------------------------------------
# 2. Cross Attention Fusion
# ------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, video, text):
        """
        video: (B, Tv, D)
        text:  (B, Tt, D)
        """
        out, _ = self.attn(query=text, key=video, value=video)
        return self.norm(out + text)


# ------------------------------------------------------------
# 3. CrossModalQA – MODEL CHÍNH
# ------------------------------------------------------------
class CrossModalQA(nn.Module):
    def __init__(
        self,
        text_encoder_name="vinai/phobert-base-v2",
        num_choices=4,
        hidden_dim=512,
        freeze_text_layers=9  # Freeze all except last 3 layers
    ):
        super().__init__()

        # -------------------------
        # TEXT ENCODER
        # -------------------------
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)

        # Freeze lower layers – improve stability & avoid all-A collapse
        if freeze_text_layers > 0:
            for name, param in self.text_encoder.named_parameters():
                if any(f"layer.{i}." in name for i in range(freeze_text_layers)):
                    param.requires_grad = False

        text_dim = self.text_encoder.config.hidden_size

        # Project text to hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # -------------------------
        # VIDEO ENCODER
        # -------------------------
        self.video_encoder = VideoEncoder(input_dim=768, hidden_dim=hidden_dim)

        # -------------------------
        # CROSS ATTENTION
        # -------------------------
        self.cross = CrossAttention(hidden_dim=hidden_dim)

        # -------------------------
        # FINAL CLASSIFIER
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_choices),
        )

    def forward(self, video_feats, input_ids, attention_mask):
        """
        video_feats: (B, T, 768)
        input_ids: (B * C, L)
        """
        B = video_feats.size(0)

        # ----------------------
        # 1. Encode text
        # ----------------------
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state  # (B*C, L, D)

        text_emb = self.text_proj(text_output)

        # reshape B*C -> (B, C, L, dim)
        C = int(text_emb.size(0) // B)
        text_emb = text_emb.view(B, C, text_emb.size(1), text_emb.size(2))

        # ----------------------
        # 2. Encode video
        # ----------------------
        video_emb = self.video_encoder(video_feats)  # (B, Tv, dim)

        # ----------------------
        # 3. Cross attention per choice
        # ----------------------
        fused = []
        for c in range(C):
            fused_c = self.cross(video_emb, text_emb[:, c])
            fused.append(fused_c.mean(dim=1))  # pool
        fused = torch.stack(fused, dim=1)  # (B, C, dim)

        # ----------------------
        # 4. Classifier
        # ----------------------
        logits = self.classifier(fused)  # (B, C, 4)

        return logits
