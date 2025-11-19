# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/ensemble_inference.py
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn_inference
from model import EarlyFusionQA
from model_improved import ImprovedVideoQA

class EnsembleModel:
    def __init__(self, model_configs, device="cuda"):
        self.models = []
        self.device = device
        
        for model_path, model_type in model_configs:
            if model_type == "early_fusion":
                model = EarlyFusionQA(text_model_name="vinai/phobert-base")
            elif model_type == "improved":
                model = ImprovedVideoQA(text_model_name="vinai/phobert-base")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            checkpoint = torch.load(model_path, map_location=device)
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
        
        # Average logits
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        return avg_logits

def ensemble_inference(model_configs, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize ensemble
    ensemble = EnsembleModel(model_configs, device)
    
    # Configuration
    APPEARANCE_DIR = "/kaggle/working/features_test/appearance"
    MOTION_DIR = "/kaggle/working/features_test/motion"
    TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
    BATCH_SIZE = 8
    
    # Load tokenizer + dataset
    print("📚 Preparing test dataset ...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    test_ds = FeatureVideoQADataset(TEST_JSON, APPEARANCE_DIR, MOTION_DIR,
                                    tokenizer_name="vinai/phobert-base", max_len=64, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_inference, num_workers=2)
    
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Inference"):
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance_feats = batch["appearance"].to(device)
            motion_feats = batch["motion"].to(device)
            ids = batch["ids"]
            
            logits = ensemble.predict(input_ids, attention_mask, appearance_feats, motion_feats)
            preds = logits.argmax(dim=1).cpu().tolist()
            
            for qid, pred in zip(ids, preds):
                results.append({"id": qid, "answer": int(pred)})
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Saved ensemble predictions to {output_file}")
    
    return results

if __name__ == "__main__":
    # Define model configurations: (model_path, model_type)
    model_configs = [
        ("/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints_advanced/best_advanced.pt", "improved"),
        ("/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints_improved_iter_1/best_self_trained.pt", "improved"),
        ("/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints_improved_iter_2/best_self_trained.pt", "improved"),
        ("/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints_improved_iter_3/best_self_trained.pt", "improved"),
    ]
    
    # Filter to existing models
    existing_configs = [config for config in model_configs if os.path.exists(config[0])]
    
    if len(existing_configs) == 0:
        print("❌ No models found for ensemble.")
    else:
        output_file = "/kaggle/working/submission_ensemble.json"
        ensemble_inference(existing_configs, output_file)