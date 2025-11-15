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
        dropout=0.3,
        num_classes=4  # max số lựa chọn
    ):
        super().__init__()

        # ----- Text encoder -----
        self.text_encoder = AutoModel.from_pretrained(text_model_name, return_dict=True)

        # ----- Projection layers -----
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.video_proj = nn.Linear(video_feat_dim, hidden_dim)
        self.ocr_proj = nn.Linear(video_feat_dim, hidden_dim)
        self.face_proj = nn.Linear(video_feat_dim, hidden_dim)

        # ----- Cross-modal Transformer -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ----- Classifier -----
        self.classifier_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.out_layer = nn.Linear(hidden_dim // 2, num_classes)
        self.num_classes = num_classes

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
        input_ids: (B, C, L)
        attention_mask: (B, C, L)
        appearance, motion: (B, T, D)
        """
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
        vid_tokens = (app_proj[:, :T, :] + mot_proj[:, :T, :]) / 2

        # ----- Repeat for C choices -----
        vid_tokens_rep = vid_tokens.unsqueeze(1).repeat(1, C, 1, 1).view(B * C, vid_tokens.size(1), -1)

        # ----- Fusion -----
        fused_seq = torch.cat([tok_emb, vid_tokens_rep], dim=1)
        fused_mask = torch.cat([
            flat_attn,
            torch.ones((B * C, vid_tokens_rep.size(1)), device=device, dtype=flat_attn.dtype)
        ], dim=1)
        src_key_padding_mask = (fused_mask == 0)
        out = self.cross_transformer(fused_seq, src_key_padding_mask=src_key_padding_mask)

        # ----- Pooling + classifier -----
        pooled = out[:, 0, :]
        h = self.classifier_layer(pooled)
        logits = self.out_layer(h)  # (B*C, num_classes)

        # ----- reshape -----
        logits = logits.view(B, C, -1)  # (B, C, num_classes)
        return logits
