import numpy as np
import os

def check_feature_dimensions():
    dirs_to_check = [
        "/kaggle/working/features_v2/appearance",
        "/kaggle/working/features_v2/motion",
        "/kaggle/working/features_test/appearance", 
        "/kaggle/working/features_test/motion"
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
            if files:
                sample_path = os.path.join(dir_path, files[0])
                sample_feat = np.load(sample_path)
                print(f"📁 {dir_path}: {sample_feat.shape}")
            else:
                print(f"📁 {dir_path}: No files found")
        else:
            print(f"📁 {dir_path}: Directory not found")

check_feature_dimensions()