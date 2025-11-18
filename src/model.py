import torch
import torch.nn as nn
from transformers import AutoModel


# ============================================================
#          VIDEO-QA MULTIPLE CHOICE MODEL (FULL FIXED)
# ============================================================

class VideoQAModel(nn.Module):
    def __init__(
        self,
        text_model_name="vinai/phobert-base-v2",
        hidden_dim=768,
        proj_dim=512,
        fusion_dim=512,
    ):
        super().__init__()

        # -----------------------------------------------------
        # Text encoder
        # -----------------------------------------------------
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Freeze most layers → unfreeze last 3
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        for layer in self.text_encoder.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        # -----------------------------------------------------
        # Visual projections
        # (appearance + motion)
        # Both features originally 768
        # -----------------------------------------------------
        self.appearance_proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
        )

        self.motion_proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
        )

        # -----------------------------------------------------
        # Fusion module (GMU)
        # A strong improvement over concat+MLP
        # -----------------------------------------------------
        self.fusion_gate = nn.Linear(proj_dim + proj_dim + hidden_dim, fusion_dim)
        self.fusion_info = nn.Linear(proj_dim + proj_dim + hidden_dim, fusion_dim)

        # -----------------------------------------------------
        # Output layer (VERY IMPORTANT)
        # 1 logit per choice
        # -----------------------------------------------------
        self.out_layer = nn.Linear(fusion_dim, 1)

    # ============================================================
    #                          FORWARD
    # ============================================================
    def forward(
        self,
        input_ids,
        attention_mask,
        appearance_feats,
        motion_feats,
    ):
        """
        input_ids:       (B*C, L)
        attention_mask:  (B*C, L)
        appearance_feats:(B*C, T, 768)
        motion_feats:    (B*C, T, 768)
        """

        B_times_C = input_ids.size(0)

        # ----------------------------
        # TEXT ENCODER
        # ----------------------------
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS embedding
        text_repr = text_outputs.last_hidden_state[:, 0, :]    # (B*C, hidden_dim)

        # ----------------------------
        # VISUAL FEATURE POOLING
        # ----------------------------
        # simple mean-pooling over time dimension
        app = appearance_feats.mean(dim=1)  # (B*C, 768)
        mot = motion_feats.mean(dim=1)      # (B*C, 768)

        # projections
        app = self.appearance_proj(app)     # (B*C, proj_dim)
        mot = self.motion_proj(mot)         # (B*C, proj_dim)

        # ----------------------------
        # CONCAT → GMU FUSION
        # ----------------------------
        fused = torch.cat([text_repr, app, mot], dim=-1)  # (B*C, hidden+proj*2)
        gate = torch.sigmoid(self.fusion_gate(fused))      # (B*C, fusion_dim)
        info = torch.tanh(self.fusion_info(fused))         # (B*C, fusion_dim)

        h = gate * info                                     # (B*C, fusion_dim)

        # ----------------------------
        # OUTPUT 1 LOGIT / CHOICE
        # ----------------------------
        logits = self.out_layer(h)  # (B*C, 1)

        return logits
