# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/convert_features.py
import os
import numpy as np
from tqdm import tqdm

def convert_features_2048_to_768(input_dir, output_dir):
    """
    Convert 2048D features to 768D using PCA or simple projection
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"🔄 Converting {len(files)} features from 2048D to 768D...")
    
    for file in tqdm(files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # Load 2048D feature
        feat_2048 = np.load(input_path)
        
        # Simple projection: take first 768 dimensions or average pooling
        if feat_2048.shape[-1] == 2048:
            # Method 1: Take first 768 dimensions
            if feat_2048.shape[0] > 0:
                feat_768 = feat_2048[:, :768]
            else:
                feat_768 = np.zeros((1, 768), dtype=np.float32)
            
            # Method 2: Alternatively, use average pooling
            # feat_768 = feat_2048.reshape(feat_2048.shape[0], 4, 512).mean(axis=1)
            
            np.save(output_path, feat_768)
        else:
            # Already correct dimension, just copy
            np.save(output_path, feat_2048)
    
    print(f"✅ Conversion complete! Output saved to: {output_dir}")

if __name__ == "__main__":
    # Convert training features
    convert_features_2048_to_768(
        "/kaggle/working/features_v2/appearance",
        "/kaggle/working/features_v2/appearance_768"
    )
    convert_features_2048_to_768(
        "/kaggle/working/features_v2/motion", 
        "/kaggle/working/features_v2/motion_768"
    )
    
    # Convert test features  
    convert_features_2048_to_768(
        "/kaggle/working/features_test/appearance",
        "/kaggle/working/features_test/appearance_768"
    )
    convert_features_2048_to_768(
        "/kaggle/working/features_test/motion",
        "/kaggle/working/features_test/motion_768"
    )