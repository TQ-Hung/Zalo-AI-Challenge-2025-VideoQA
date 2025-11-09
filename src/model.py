# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class CrossModalQA(nn.Module):
    def __init__(self, text_model_name="vinai/phobert-base-v2",
             video_feat_dim=768, text_dim=768, hidden_dim=512,
             n_heads=8, n_layers=3, dropout=0.2, text_pooling="cls"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name, return_dict=True)

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.video_proj = nn.Linear(video_feat_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.cross_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, input_ids, attention_mask, appearance, motion):
        """
        input_ids: (B, C, L)
        attention_mask: (B, C, L)
        appearance, motion: (B, T, D) hoặc (B*C, T, D)
        """
        B, C, L = input_ids.shape
        device = input_ids.device

        # ----- Encode text -----
        flat_input_ids = input_ids.view(B * C, L)
        flat_attn = attention_mask.view(B * C, L)
        text_out = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attn)
        tok_emb = self.text_proj(text_out.last_hidden_state)  # (B*C, L, hidden_dim)

        # ----- Video embedding -----
        # có thể đã được flatten (B*C, T, D)
        if appearance.dim() == 3 and appearance.size(0) == B:
            # chưa flatten → (B, T, D)
            app_proj = self.video_proj(appearance)  # (B, T, H)
            mot_proj = self.video_proj(motion)
            T = min(app_proj.size(1), mot_proj.size(1))
            vid_tokens = (app_proj[:, :T, :] + mot_proj[:, :T, :]) / 2
            vid_tokens_rep = vid_tokens.unsqueeze(1).repeat(1, C, 1, 1).view(B * C, T, -1)
        else:
            # đã flatten
            app_proj = self.video_proj(appearance)
            mot_proj = self.video_proj(motion)
            T = min(app_proj.size(1), mot_proj.size(1))
            vid_tokens_rep = (app_proj[:, :T, :] + mot_proj[:, :T, :]) / 2  # (B*C, T, H)

        # ----- Fusion -----
        # kiểm tra dimensions
        assert tok_emb.dim() == 3, f"tok_emb wrong shape: {tok_emb.shape}"
        assert vid_tokens_rep.dim() == 3, f"vid_tokens_rep wrong shape: {vid_tokens_rep.shape}"

        fused_seq = torch.cat([tok_emb, vid_tokens_rep], dim=1)  # (B*C, L+T, H)

        fused_mask = torch.cat([
            flat_attn,
            torch.ones((B * C, vid_tokens_rep.size(1)), device=device, dtype=flat_attn.dtype)
        ], dim=1)
        src_key_padding_mask = (fused_mask == 0)

        out = self.cross_transformer(fused_seq, src_key_padding_mask=src_key_padding_mask)
        pooled = out[:, 0, :]
        logits = self.classifier(pooled).view(B, C)
        return logits
