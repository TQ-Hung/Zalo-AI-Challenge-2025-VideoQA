# src/iterative_self_training.py
import os
import json
from pseudo_labeling import generate_pseudo_labels
from train_self_training import train_with_pseudo_labels

def iterative_self_training(num_iterations=3, initial_threshold=0.8, threshold_decay=0.1):
    """
    Perform iterative self-training with decreasing confidence thresholds
    """
    # Configuration
    BASE_MODEL = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
    TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
    APPEARANCE_DIR = "features_test/appearance"
    MOTION_DIR = "features_test/motion"
    ORIGINAL_TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
    
    current_model = BASE_MODEL
    confidence_threshold = initial_threshold
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"🚀 Starting Self-Training Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Step 1: Generate pseudo-labels with current model
        pseudo_output = f"/kaggle/working/pseudo_labels_iter_{iteration+1}.json"
        print(f"📝 Step 1: Generating pseudo-labels with threshold {confidence_threshold:.2f}")
        
        generate_pseudo_labels(
            model_checkpoint=current_model,
            test_json=TEST_JSON,
            appearance_dir=APPEARANCE_DIR,
            motion_dir=MOTION_DIR,
            output_file=pseudo_output,
            confidence_threshold=confidence_threshold
        )
        
        # Step 2: Train new model with combined data
        print(f"🎯 Step 2: Training with combined dataset")
        
        # Update the pseudo_labels path for training
        import train_self_training
        train_self_training.PSEUDO_LABELS_JSON = pseudo_output
        train_self_training.OUTPUT_DIR = f"checkpoints_iter_{iteration+1}"
        train_self_training.PSEUDO_WEIGHT = 0.7 - (iteration * 0.2)  # Decrease weight each iteration
        
        # Train the model
        train_self_training.train_with_pseudo_labels()
        
        # Update current model to the newly trained one
        current_model = f"checkpoints_iter_{iteration+1}/best_self_trained.pt"
        
        # Decrease threshold for next iteration
        confidence_threshold = max(0.5, confidence_threshold - threshold_decay)
        
        print(f"✅ Iteration {iteration + 1} completed!")
        print(f"📁 New model: {current_model}")
        print(f"🎯 Next threshold: {confidence_threshold:.2f}")
    
    print(f"\n🎉 All {num_iterations} iterations completed!")
    return current_model

def create_final_ensemble(model_paths, output_path="checkpoints_ensemble/final_ensemble.pt"):
    """
    Create a final ensemble model from all iterations
    """
    import torch
    from model import EarlyFusionQA
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create ensemble by averaging weights (simple approach)
    print("🤝 Creating ensemble model...")
    
    # Load first model to get architecture
    first_model = EarlyFusionQA(text_model_name="vinai/phobert-base")
    first_ckpt = torch.load(model_paths[0], map_location=device)
    
    # Handle DataParallel
    if "module" in list(first_ckpt["model"].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in first_ckpt["model"].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        first_model.load_state_dict(new_state_dict)
    else:
        first_model.load_state_dict(first_ckpt["model"])
    
    # Initialize average state dict
    avg_state_dict = first_model.state_dict()
    
    # Average weights from all models
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] * (1.0 / len(model_paths))
    
    for i, model_path in enumerate(model_paths[1:], 1):
        model = EarlyFusionQA(text_model_name="vinai/phobert-base")
        ckpt = torch.load(model_path, map_location=device)
        
        if "module" in list(ckpt["model"].keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt["model"].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(ckpt["model"])
        
        state_dict = model.state_dict()
        for key in avg_state_dict.keys():
            avg_state_dict[key] += state_dict[key] * (1.0 / len(model_paths))
    
    # Save ensemble model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "model": avg_state_dict,
        "ensemble_sources": model_paths,
        "info": "Ensemble of self-training iterations"
    }, output_path)
    
    print(f"✅ Ensemble model saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Run 3 iterations of self-training
    final_model = iterative_self_training(
        num_iterations=3,
        initial_threshold=0.8,
        threshold_decay=0.1
    )
    
    # Create ensemble from all iterations
    model_paths = [
        "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt",  # Original
        "checkpoints_iter_1/best_self_trained.pt",
        "checkpoints_iter_2/best_self_trained.pt", 
        "checkpoints_iter_3/best_self_trained.pt"
    ]
    
    # Filter to only existing paths
    existing_paths = [p for p in model_paths if os.path.exists(p)]
    if len(existing_paths) >= 2:
        ensemble_path = create_final_ensemble(existing_paths)
        print(f"🎉 Final ensemble created: {ensemble_path}")
    else:
        print("⚠️ Not enough models for ensemble, using the last self-trained model")