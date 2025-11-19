# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/model_improved.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import math

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=mask)
        x = self.layer_norm(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm(x + self.dropout(ffn_output))
        return x

class CrossModalFusion(nn.Module):
    def __init__(self, text_dim=768, video_dim=512, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features, video_features):
        # text_features: (B*C, hidden_dim)
        # video_features: (B, T, hidden_dim)
        B, T, D = video_features.shape
        C = text_features.shape[0] // B
        
        # Repeat video features for each choice
        video_features = video_features.unsqueeze(1).repeat(1, C, 1, 1).view(B*C, T, D)
        text_features = text_features.unsqueeze(1)  # (B*C, 1, D)
        
        # Cross attention: text as query, video as key/value
        fused, _ = self.cross_attn(text_features, video_features, video_features)
        fused = self.layer_norm(text_features + self.dropout(fused))
        return fused.squeeze(1)

class ImprovedVideoQA(nn.Module):
    def __init__(self, 
                 text_model_name="vinai/phobert-base",
                 video_dim=768,
                 text_dim=768,
                 hidden_dim=512,
                 num_layers=2,
                 num_heads=8,
                 dropout=0.2):
        super().__init__()
        
        # Text encoder with gradient checkpointing
        self.text_encoder = AutoModel.from_pretrained(text_model_name, return_dict=True)
        text_config = AutoConfig.from_pretrained(text_model_name)
        
        # Video encoder with multi-scale temporal attention
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.temporal_layers = nn.ModuleList([
            MultiScaleTemporalAttention(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.video_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(text_dim, hidden_dim, hidden_dim, num_heads, dropout)
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, attention_mask, appearance, motion):
        B, C, L = input_ids.shape
        device = input_ids.device

        # Text encoding
        flat_input_ids = input_ids.view(B * C, L)
        flat_attention_mask = attention_mask.view(B * C, L)
        text_out = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        text_features = text_out.last_hidden_state[:, 0]  # CLS token

        # Video encoding with multi-scale processing
        T = max(appearance.size(1), motion.size(1))
        if appearance.size(1) < T:
            appearance = torch.cat([appearance, torch.zeros(B, T - appearance.size(1), appearance.size(2), device=device)], dim=1)
        if motion.size(1) < T:
            motion = torch.cat([motion, torch.zeros(B, T - motion.size(1), motion.size(2), device=device)], dim=1)

        video_seq = (appearance + motion) / 2
        video_features = self.video_proj(video_seq)
        
        # Apply temporal layers
        for layer in self.temporal_layers:
            video_features = layer(video_features)
        
        video_features = self.video_norm(video_features.mean(dim=1))  # Global temporal pooling

        # Cross-modal fusion
        fused_features = self.cross_modal_fusion(text_features, video_features.unsqueeze(1))

        # Classification
        logits = self.classifier(fused_features).view(B, C)
        return logits