# /kaggle/working/Zalo-AI-Challenge-2025-VideoQA/src/data_augmentation.py
import numpy as np
import torch

class FeatureAugmentation:
    def __init__(self, aug_prob=0.3):
        self.aug_prob = aug_prob
    
    def temporal_masking(self, features, mask_ratio=0.1):
        """Randomly mask temporal segments"""
        if np.random.random() < self.aug_prob:
            T, D = features.shape
            mask_len = max(1, int(T * mask_ratio))
            start_idx = np.random.randint(0, T - mask_len)
            features[start_idx:start_idx + mask_len] = 0
        return features
    
    def feature_noise(self, features, noise_std=0.01):
        """Add Gaussian noise to features"""
        if np.random.random() < self.aug_prob:
            noise = torch.randn_like(features) * noise_std
            features = features + noise
        return features
    
    def random_scale(self, features, scale_range=(0.9, 1.1)):
        """Randomly scale features"""
        if np.random.random() < self.aug_prob:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            features = features * scale
        return features
    
    def apply_augmentation(self, appearance, motion):
        appearance = self.temporal_masking(appearance)
        appearance = self.feature_noise(appearance)
        appearance = self.random_scale(appearance)
        
        motion = self.temporal_masking(motion)
        motion = self.feature_noise(motion)
        motion = self.random_scale(motion)
        
        return appearance, motion