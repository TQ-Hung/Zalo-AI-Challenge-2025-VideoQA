# FILE: src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class VideoEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, video_feats):
        # video_feats: (B, T, 768)
        x = self.proj(video_feats)
        x, _ = self.gru(x)
        x = self.norm(x)
        return x  # (B, T, hidden_dim)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, video, text):
        # video: (B, Tv, D), text: (B, Tt, D)
        out, _ = self.attn(query=text, key=video, value=video)
        return self.norm(out + text)


class CrossModalQA(nn.Module):
    def __init__(
        self,
        text_encoder_name="vinai/phobert-base-v2",
        num_choices=4,
        hidden_dim=512,
        freeze_text_layers=9,
    ):
        super().__init__()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)

        # Optionally freeze lower layers for stability
        if freeze_text_layers > 0 and hasattr(self.text_encoder, "encoder"):
            for name, param in self.text_encoder.named_parameters():
                if any(f"layer.{i}." in name for i in range(freeze_text_layers)):
                    param.requires_grad = False

        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Video encoder
        self.video_encoder = VideoEncoder(input_dim=768, hidden_dim=hidden_dim)

        # Cross-attention
        self.cross = CrossAttention(hidden_dim=hidden_dim, num_heads=8)

        # Classifier -> 1 score per choice
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self.num_choices = num_choices

    def forward(self, video_feats, input_ids, attention_mask):
        """
        video_feats: (B, Tv, 768)
        input_ids: (B, C, L) or (B*C, L)
        attention_mask: same shape as input_ids
        Returns logits shape (B, C)
        """
        # If inputs are (B, C, L) flatten to (B*C, L) for text encoder
        if input_ids.dim() == 3:
            B, C, L = input_ids.shape
            input_ids_flat = input_ids.view(B * C, L)
            attention_mask_flat = attention_mask.view(B * C, L)
        else:
            # assume already flattened
            input_ids_flat = input_ids
            attention_mask_flat = attention_mask
            # need to infer B and C for reshape later
            B = video_feats.size(0)
            C = int(input_ids_flat.size(0) // B)

        # encode text
        text_output = self.text_encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
        ).last_hidden_state  # (B*C, L, D)

        text_emb = self.text_proj(text_output)  # (B*C, L, hidden_dim)

        # encode video once (per example)
        video_emb = self.video_encoder(video_feats)  # (B, Tv, hidden_dim)

        # Expand video to align with choices: (B, Tv, D) -> (B*C, Tv, D)
        video_emb_expand = video_emb.unsqueeze(1).expand(B, C, video_emb.size(1), video_emb.size(2))
        video_emb_flat = video_emb_expand.contiguous().view(B * C, video_emb.size(1), video_emb.size(2))

        # cross attend per flattened item
        fused = self.cross(video_emb_flat, text_emb)  # (B*C, L, D)

        # pooled representation per (B*C)
        pooled = fused.mean(dim=1)  # (B*C, D)

        logits_flat = self.classifier(pooled)  # (B*C, 1)
        logits = logits_flat.view(B, C)  # (B, C)

        return logits


