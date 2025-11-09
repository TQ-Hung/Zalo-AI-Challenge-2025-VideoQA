# src/model.py  (replace or add class CrossModalQA)
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import RobertaTokenizer
class CrossModalQA(nn.Module):
    def __init__(self, text_model_name="vinai/phobert-base-v2",
                 video_feat_dim=768, text_dim=768, hidden_dim=512,
                 n_heads=8, n_layers=3, dropout=0.2, text_pooling="cls"):
        super().__init__()
        # text encoder (pretrained)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, return_dict=True)
        self.text_pooling = text_pooling
        # project text tokens to hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # project frame features to hidden_dim
        self.video_proj = nn.Linear(video_feat_dim, hidden_dim)

        # build cross-modal transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                                   dim_feedforward=hidden_dim*4,
                                                   dropout=dropout, batch_first=True)
        self.cross_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # classifier head: produce score per choice
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def pool_text(self, out, mask):
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        # mean pooling over CLS? here we keep token outputs for cross-attn
        return out.last_hidden_state  # (B*C, L, text_dim)

    def forward(self, input_ids, attention_mask, appearance, motion):
        """
        input_ids: (B, C, L)
        attention_mask: (B, C, L)
        appearance: (B, T_app, D_app)
        motion: (B, T_mot, D_mot)
        """
        B, C, L = input_ids.shape
        device = input_ids.device

        # --- text tokens ---
        flat_input_ids = input_ids.view(B * C, L)
        flat_attn = attention_mask.view(B * C, L)
        text_out = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attn)
        tok_emb = self.pool_text(text_out, flat_attn)   # (B*C, L, text_dim)
        tok_emb = self.text_proj(tok_emb)               # (B*C, L, hidden_dim)

        # --- video tokens: create frame tokens ---
        # unify appearance & motion length by padding/truncation in collate_fn.
        # here we assume appearance and motion are (B, T, D)
        # simple fusion: concat along feature dim and project
        # choose T = min(T_app, T_mot) or max with padding handled earlier
        if appearance.dim() == 3 and motion.dim() == 3:
            # project appearance and motion separately then sum
            app_p = self.video_proj(appearance)  # (B, T_app, hidden_dim)
            mot_p = self.video_proj(motion)      # (B, T_mot, hidden_dim)
            # if lengths differ we pad earlier; for simplicity take mean of two
            # align to same T by taking first min(T_app,T_mot)
            T = min(app_p.size(1), mot_p.size(1))
            vid_tokens = (app_p[:, :T, :] + mot_p[:, :T, :]) / 2.0
        else:
            # fallback: appearance already (B, T, hidden_dim)
            vid_tokens = self.video_proj(appearance)

        # repeat video tokens per choice
        vid_tokens_rep = vid_tokens.unsqueeze(1).repeat(1, C, 1, 1).view(B*C, vid_tokens.size(1), -1)  # (B*C, T, hidden_dim)

        # --- combine text tokens and video tokens into single sequence ---
        # concat along sequence dimension: [text_tokens ; video_tokens]
        vid_tokens_rep = vid_tokens_rep.unsqueeze(1)  # (B*C, 1, hidden_dim)
        fused_seq = torch.cat([tok_emb, vid_tokens_rep], dim=1)  # (B*C, L+1, hidden_dim)

        # optionally create attention mask for cross transformer:
        # text_mask + video_mask (video masked ones are ones)
        text_mask = flat_attn
        video_mask = torch.ones((B*C, vid_tokens_rep.size(1)), device=device, dtype=torch.long)
        fused_mask = torch.cat([text_mask, video_mask], dim=1)   # (B*C, L+T)

        # transformer expects float mask? simpler: create src_key_padding_mask with True for PAD
        src_key_padding_mask = (fused_mask == 0)  # True where padding

        # pass through cross modal transformer
        out = self.cross_transformer(fused_seq, src_key_padding_mask=src_key_padding_mask)  # (B*C, L+T, hidden_dim)

        # pool (take first token -> CLS of text portion)
        pooled = out[:, 0, :]  # (B*C, hidden_dim)
        logits = self.classifier(pooled).view(B, C)  # (B, C)
        return logits
