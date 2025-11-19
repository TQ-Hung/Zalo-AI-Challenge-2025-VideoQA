# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/pseudo_labeling.py
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn_inference
from model import EarlyFusionQA

def generate_pseudo_labels(model_checkpoint, test_json, appearance_dir, motion_dir, output_file, confidence_threshold=0.7):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting pseudo-label generation with confidence threshold: {confidence_threshold}")
    
    # Auto-detect feature dimension by checking first feature file
    feature_dim = 768  # default
    try:
        sample_files = [f for f in os.listdir(appearance_dir) if f.endswith('.npy')]
        if sample_files:
            sample_path = os.path.join(appearance_dir, sample_files[0])
            sample_feat = np.load(sample_path)
            feature_dim = sample_feat.shape[-1]
            print(f"🔍 Detected feature dimension: {feature_dim}")
    except Exception as e:
        print(f"⚠️ Could not detect feature dimension, using default: {feature_dim}")
    
    # Load model with correct dimension
    model = EarlyFusionQA(text_model_name="vinai/phobert-base", video_dim=feature_dim).to(device)
    ckpt = torch.load(model_checkpoint, map_location=device)
    
    # Handle dimension mismatch in state_dict
    state_dict = ckpt["model"]
    
    # Check if there's dimension mismatch
    if "video_proj.weight" in state_dict:
        expected_shape = state_dict["video_proj.weight"].shape
        current_shape = model.video_proj.weight.shape
        if expected_shape != current_shape:
            print(f"🔄 Adjusting video_proj layer: {expected_shape} -> {current_shape}")
            # Remove mismatched layers, they will be randomly initialized
            keys_to_remove = [k for k in state_dict.keys() if k.startswith(('video_proj', 'temporal_agg'))]
            for k in keys_to_remove:
                state_dict.pop(k)
            print(f"✅ Removed mismatched layers: {keys_to_remove}")
    
    # Handle DataParallel wrapping
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
    
    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    test_ds = FeatureVideoQADataset(
        test_json, 
        appearance_dir, 
        motion_dir,
        tokenizer_name="vinai/phobert-base", 
        max_len=64, 
        is_test=True
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=collate_fn_inference, num_workers=2
    )
    
    pseudo_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating pseudo-labels"):
            if batch is None:
                continue
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance = batch["appearance"].to(device)
            motion = batch["motion"].to(device)
            ids = batch["ids"]
            
            logits = model(input_ids, attention_mask, appearance, motion)
            probabilities = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)
            
            for i, (qid, pred, prob) in enumerate(zip(ids, preds.cpu(), max_probs.cpu())):
                # Find the original item to get question and choices
                original_item = None
                for item in test_ds.items:
                    if item["id"] == qid:
                        original_item = item
                        break
                
                if original_item and prob >= confidence_threshold:
                    # Get the actual answer text from choices
                    answer_text = original_item["choices"][pred.item()]
                    
                    pseudo_labels.append({
                        "id": qid,
                        "video_path": original_item["video_path"],
                        "question": original_item["question"],
                        "choices": original_item["choices"],
                        "answer": answer_text,
                        "answer_index": int(pred),
                        "confidence": float(prob)
                    })
    
    # Save pseudo-labels in the same format as training data
    output_data = {
        "data": pseudo_labels,
        "info": {
            "generated_by": "pseudo_labeling",
            "confidence_threshold": confidence_threshold,
            "total_samples": len(pseudo_labels),
            "feature_dim": feature_dim
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated {len(pseudo_labels)} pseudo-labels with confidence ≥ {confidence_threshold}")
    print(f"📁 Saved to: {output_file}")
    return pseudo_labels

if __name__ == "__main__":
    # Configuration - UPDATE PATHS FOR KAGGLE
    MODEL_CHECKPOINT = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
    TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
    APPEARANCE_DIR = "/kaggle/working/features_test/appearance"
    MOTION_DIR = "/kaggle/working/features_test/motion"
    OUTPUT_FILE = "/kaggle/working/pseudo_labels.json"
    
    # Create directories if not exist
    os.makedirs("/kaggle/working/features_test/appearance", exist_ok=True)
    os.makedirs("/kaggle/working/features_test/motion", exist_ok=True)
    
    generate_pseudo_labels(
        model_checkpoint=MODEL_CHECKPOINT,
        test_json=TEST_JSON,
        appearance_dir=APPEARANCE_DIR,
        motion_dir=MOTION_DIR,
        output_file=OUTPUT_FILE,
        confidence_threshold=0.7
    )