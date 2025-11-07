# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class TemporalAggregator(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512):
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

class EarlyFusionQA(nn.Module):
    """
    EarlyFusionQA + Temporal Aggregation (BiGRU)
    - Text: BERT encoder -> pooled representation
    - Video: BiGRU over frame sequences (appearance + motion)
    - Fusion: gated residual + transformer fusion
    """
    def __init__(self, 
                 text_model_name="bert-base-multilingual-cased",
                 video_dim=2048,
                 text_dim=768,
                 hidden_dim=512,
                 fusion_dim=512,
                 dropout=0.2,
                 text_pooling="cls"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name, return_dict=True)
        self.text_pooling = text_pooling
        self.video_dropout = nn.Dropout(dropout)

        # --- Temporal projection + GRU for video ---
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.video_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_agg = TemporalAggregator(in_dim=hidden_dim, hidden_dim=hidden_dim // 2)
        # self.video_norm = nn.LayerNorm(hidden_dim)

        # --- Text projection ---
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Fusion ---
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )

    def pool_text(self, out, mask):
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
        appearance: (B, T_app, 2048)
        motion: (B, T_mot, 2048)
        """
        B, C, L = input_ids.shape
        device = input_ids.device

        # --- Text encoding ---
        flat_input_ids = input_ids.view(B * C, L)
        flat_attention_mask = attention_mask.view(B * C, L)
        text_out = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled = self.pool_text(text_out, flat_attention_mask)
        text_feat = self.text_proj(pooled)  # (B*C, hidden_dim)

        # --- Video temporal encoding ---
        # Combine app & motion sequences (pad shorter one)
        T = max(appearance.size(1), motion.size(1))
        if appearance.size(1) < T:
            pad = torch.zeros(B, T - appearance.size(1), appearance.size(2), device=device)
            appearance = torch.cat([appearance, pad], dim=1)
        if motion.size(1) < T:
            pad = torch.zeros(B, T - motion.size(1), motion.size(2), device=device)
            motion = torch.cat([motion, pad], dim=1)

        video_seq = (appearance + motion) / 2  # simple fusion
        video_seq = self.video_proj(video_seq)
        video_out, _ = self.video_gru(video_seq)
        video_feat = self.temporal_agg(video_out)

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
