# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/inference_self_trained.py
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn_inference
from model import EarlyFusionQA

def inference_self_trained(model_path, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect feature dimension
    feature_dim = 768  # default
    try:
        appearance_dir = "/kaggle/working/features_test/appearance_768"
        sample_files = [f for f in os.listdir(appearance_dir) if f.endswith('.npy')]
        if sample_files:
            sample_path = os.path.join(appearance_dir, sample_files[0])
            sample_feat = np.load(sample_path)
            feature_dim = sample_feat.shape[-1]
            print(f"🔍 Detected feature dimension: {feature_dim}")
    except Exception as e:
        print(f"⚠️ Could not detect feature dimension, using default: {feature_dim}")
    
    # Load model
    print(f"🤖 Loading self-trained model from {model_path} ...")
    model = EarlyFusionQA(text_model_name="vinai/phobert-base", video_dim=feature_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)
    
    # Handle dimension mismatch
    state_dict = ckpt["model"]
    
    # Check if there's dimension mismatch
    if "video_proj.weight" in state_dict:
        expected_shape = state_dict["video_proj.weight"].shape
        current_shape = model.video_proj.weight.shape
        if expected_shape != current_shape:
            print(f"🔄 Adjusting video_proj layer: {expected_shape} -> {current_shape}")
            # Remove mismatched layers
            keys_to_remove = [k for k in state_dict.keys() if k.startswith(('video_proj', 'temporal_agg'))]
            for k in keys_to_remove:
                state_dict.pop(k)
            print(f"✅ Removed mismatched layers: {keys_to_remove}")
    
    # Handle DataParallel
    if "module" in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Configuration
    APPEARANCE_DIR = "/kaggle/working/features_test/appearance_768appearance"
    MOTION_DIR = "/kaggle/working/features_test/motion_768"
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
        for batch in tqdm(test_loader, desc="Inference with Self-Trained Model"):
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance_feats = batch["appearance"].to(device)
            motion_feats = batch["motion"].to(device)
            ids = batch["ids"]
            
            logits = model(input_ids, attention_mask, appearance_feats, motion_feats)
            preds = logits.argmax(dim=1).cpu().tolist()
            
            for qid, pred in zip(ids, preds):
                results.append({"id": qid, "answer": int(pred)})
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Saved self-trained predictions to {output_file}")
    
    return results

if __name__ == "__main__":
    # Choose which model to use for inference
    MODEL_PATHS = [
        "/kaggle/working/checkpoints_self_training/best_self_trained.pt",  # Self-trained model
        "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"  # Original model
    ]
    
    # Use the first available model
    model_to_use = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_to_use = path
            print(f"🎯 Using model: {path}")
            break
    
    if model_to_use:
        output_file = f"/kaggle/working/submission_self_trained.json"
        inference_self_trained(model_to_use, output_file)
    else:
        print("❌ No self-trained model found. Please run self-training first.")