# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/pseudo_labeling_improved.py
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn_inference
from model_improved import ImprovedVideoQA

def generate_pseudo_labels_improved(model_checkpoint, test_json, appearance_dir, motion_dir, output_file, confidence_threshold=0.7):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting pseudo-label generation with ImprovedVideoQA and confidence threshold: {confidence_threshold}")
    
    # Load model
    model = ImprovedVideoQA(text_model_name="vinai/phobert-base").to(device)
    ckpt = torch.load(model_checkpoint, map_location=device)
    
    # Handle DataParallel wrapping
    if "module" in list(ckpt["model"].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt["model"].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(ckpt["model"])
    
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
            "generated_by": "pseudo_labeling_improved",
            "confidence_threshold": confidence_threshold,
            "total_samples": len(pseudo_labels)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated {len(pseudo_labels)} pseudo-labels with confidence ≥ {confidence_threshold}")
    print(f"📁 Saved to: {output_file}")
    return pseudo_labels