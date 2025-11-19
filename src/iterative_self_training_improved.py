# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/iterative_self_training_improved.py
import os
import json
import torch
from pseudo_labeling_improved import generate_pseudo_labels_improved
from train_self_training_improved import train_with_pseudo_labels_improved

def iterative_self_training_improved(num_iterations=3, initial_threshold=0.8, threshold_decay=0.1):
    """
    Perform iterative self-training with decreasing confidence thresholds using ImprovedVideoQA
    """
    # Configuration
    BASE_MODEL = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints_advanced/best_advanced.pt"
    TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
    APPEARANCE_DIR = "/kaggle/working/features_test/appearance"
    MOTION_DIR = "/kaggle/working/features_test/motion"
    ORIGINAL_TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
    
    current_model = BASE_MODEL
    confidence_threshold = initial_threshold
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"🚀 Starting Self-Training Iteration {iteration + 1}/{num_iterations} with Improved Model")
        print(f"{'='*60}")
        
        # Step 1: Generate pseudo-labels with current model
        pseudo_output = f"/kaggle/working/pseudo_labels_improved_iter_{iteration+1}.json"
        print(f"📝 Step 1: Generating pseudo-labels with threshold {confidence_threshold:.2f}")
        
        generate_pseudo_labels_improved(
            model_checkpoint=current_model,
            test_json=TEST_JSON,
            appearance_dir=APPEARANCE_DIR,
            motion_dir=MOTION_DIR,
            output_file=pseudo_output,
            confidence_threshold=confidence_threshold
        )
        
        # Step 2: Train new model with combined data
        print(f"🎯 Step 2: Training with combined dataset")
        
        # We'll pass the paths as arguments to the training function
        train_with_pseudo_labels_improved(
            original_train_json=ORIGINAL_TRAIN_JSON,
            pseudo_labels_json=pseudo_output,
            output_dir=f"checkpoints_improved_iter_{iteration+1}",
            pseudo_weight=0.7 - (iteration * 0.2)
        )
        
        # Update current model to the newly trained one
        current_model = f"checkpoints_improved_iter_{iteration+1}/best_self_trained.pt"
        
        # Decrease threshold for next iteration
        confidence_threshold = max(0.5, confidence_threshold - threshold_decay)
        
        print(f"✅ Iteration {iteration + 1} completed!")
        print(f"📁 New model: {current_model}")
        print(f"🎯 Next threshold: {confidence_threshold:.2f}")
    
    print(f"\n🎉 All {num_iterations} iterations completed!")
    return current_model

if __name__ == "__main__":
    # Run 3 iterations of self-training with improved model
    final_model = iterative_self_training_improved(
        num_iterations=3,
        initial_threshold=0.8,
        threshold_decay=0.1
    )