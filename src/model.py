# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class CrossModalQA(nn.Module):
    def __init__(
        self,
        text_model_name="vinai/phobert-base-v2",
        video_feat_dim=768,
        text_dim=768,
        hidden_dim=512,
        n_heads=8,
        n_layers=3,
        dropout=0.2,
        text_pooling="cls",
    ):
        super().__init__()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name, return_dict=True)

        # Projection layers
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.video_proj = nn.Linear(video_feat_dim, hidden_dim)
        self.ocr_proj = nn.Linear(video_feat_dim, hidden_dim)
        self.face_proj = nn.Linear(video_feat_dim, hidden_dim)

        # Cross-modal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        appearance,
        motion,
        ocr_feat=None,
        face_feat=None,
    ):
        """
        Supports 2-input or 4-input mode automatically.
        input_ids: (B, L) or (B, C, L)
        attention_mask: (B, L) or (B, C, L)
        appearance, motion, ocr_feat, face_feat: (B, T, D)
        """

        # Handle batch dimension for text
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
        B, C, L = input_ids.shape
        device = input_ids.device

        # ----- Text encoding -----
        flat_input_ids = input_ids.view(B * C, L)
        flat_attn = attention_mask.view(B * C, L)
        text_out = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attn)
        tok_emb = self.text_proj(text_out.last_hidden_state)  # (B*C, L, H)

        # ----- Video encoding -----
        app_proj = self.video_proj(appearance)
        mot_proj = self.video_proj(motion)
        T = min(app_proj.size(1), mot_proj.size(1))
        vid_tokens = (app_proj[:, :T, :] + mot_proj[:, :T, :]) / 2  # (B, T, H)

        # Optional features
        if ocr_feat is not None and face_feat is not None:
            ocr_proj = self.ocr_proj(ocr_feat)
            face_proj = self.face_proj(face_feat)
            T_feat = min(ocr_proj.size(1), face_proj.size(1))
            extra_tokens = (ocr_proj[:, :T_feat, :] + face_proj[:, :T_feat, :]) / 2
            vid_tokens = torch.cat([vid_tokens, extra_tokens], dim=1)

        # Repeat for C if needed
        vid_tokens_rep = vid_tokens.unsqueeze(1).repeat(1, C, 1, 1).view(B * C, vid_tokens.size(1), -1)

        # ----- Fusion -----
        fused_seq = torch.cat([tok_emb, vid_tokens_rep], dim=1)
        fused_mask = torch.cat([
            flat_attn,
            torch.ones((B * C, vid_tokens_rep.size(1)), device=device, dtype=flat_attn.dtype)
        ], dim=1)
        src_key_padding_mask = (fused_mask == 0)

        out = self.cross_transformer(fused_seq, src_key_padding_mask=src_key_padding_mask)

        # Pooling and classification
        pooled = out[:, 0, :]
        logits = self.classifier(pooled).view(B, C)
        return logits
