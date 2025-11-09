# compute_mean_features.py
import numpy as np
import os

APPEARANCE_DIR = "features_v2/appearance"
MOTION_DIR = "features_v2/motion"

app_feats = []
mot_feats = []

for f in os.listdir(APPEARANCE_DIR):
    if f.endswith(".npy"):
        arr = np.load(os.path.join(APPEARANCE_DIR, f))
        if arr.shape == (16, 768):
            app_feats.append(arr.mean(0))  # mean over time

for f in os.listdir(MOTION_DIR):
    if f.endswith(".npy"):
        arr = np.load(os.path.join(MOTION_DIR, f))
        if arr.shape == (16, 768):
            mot_feats.append(arr.mean(0))

mean_app = np.mean(app_feats, axis=0)
mean_mot = np.mean(mot_feats, axis=0)

np.save("mean_appearance.npy", mean_app)
np.save("mean_motion.npy", mean_mot)
print("Saved mean features!")