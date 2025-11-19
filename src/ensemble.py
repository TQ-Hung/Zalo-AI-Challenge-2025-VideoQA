# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/ensemble.py
import torch
import numpy as np
from model import EarlyFusionQA
from model_improved import ImprovedVideoQA

class EnsembleModel:
    def __init__(self, model_paths, model_types, device="cuda"):
        self.models = []
        self.device = device
        
        for path, model_type in zip(model_paths, model_types):
            if model_type == "early_fusion":
                model = EarlyFusionQA(text_model_name="vinai/phobert-base")
            elif model_type == "improved":
                model = ImprovedVideoQA(text_model_name="vinai/phobert-base")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            checkpoint = torch.load(path, map_location=device)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            self.models.append(model)
    
    def predict(self, input_ids, attention_mask, appearance, motion):
        all_logits = []
        
        for model in self.models:
            with torch.no_grad():
                logits = model(input_ids, attention_mask, appearance, motion)
                all_logits.append(logits)
        
        # Weighted average based on validation performance
        weights = [1.0] * len(self.models)  # Can be tuned based on val performance
        weighted_logits = sum(w * logits for w, logits in zip(weights, all_logits))
        
        return weighted_logits

def create_ensemble_predictions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define model paths and types
    model_configs = [
        ("/kaggle/working/checkpoints/best.pt", "early_fusion"),
        ("/kaggle/working/checkpoints_self_training/best_self_trained.pt", "early_fusion"),
        ("/kaggle/working/checkpoints_advanced/best_advanced.pt", "improved"),
    ]
    
    # Filter to existing models
    existing_configs = [(path, mtype) for path, mtype in model_configs if os.path.exists(path)]
    
    if len(existing_configs) < 2:
        print("⚠️ Not enough models for ensemble")
        return None
    
    ensemble = EnsembleModel(
        [config[0] for config in existing_configs],
        [config[1] for config in existing_configs],
        device
    )
    
    return ensemble