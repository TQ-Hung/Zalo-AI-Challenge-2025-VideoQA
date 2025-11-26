📘 Zalo AI Challenge 2025 – Video Question Answering (VideoQA)

This repository contains my full pipeline and solution for the Zalo AI Challenge 2025 – VideoQA track.
The goal of the task is to build a model that answers multiple-choice questions based on short videos, requiring multi-modal reasoning across visual features, motion features, and textual question understanding.

🚀 Features
🔹 End-to-End VideoQA Pipeline

Processing raw video features (appearance & motion)

Dataset loading, padding, batching, and augmentation

Question tokenization with transformer-based encoders

🔹 Model Architectures

Includes multiple versions of cross-modal models:

model.py – Baseline Cross-Modal QA

model_improved.py – Enhanced architecture with better fusion and deeper attention

Support for self-supervised and semi-supervised training

🔹 Training Pipelines

Provided training scripts:

train.py – Standard supervised training

train_advanced.py – Advanced training with improved optimization

train_self_training.py & improved variants

fine_tune_focal.py – Training using focal loss for imbalanced answers

🔹 Self-Training / Pseudo-Labeling

Implements weak-supervision approaches:

pseudo_labeling.py

pseudo_labeling_improved.py

iterative_self_training.py

🔹 Inference

Multiple inference strategies:

inference.py – Standard inference

ensemble_inference.py – Voting / averaging multiple models

convert_submit.py – Convert raw predictions to submission format

🧠 Approach Overview

Feature Extraction

Extract appearance (RGB) features

Extract motion (optical flow / slowfast features)

Cross-Modal Fusion Model

Text encoder (PhoBERT / BERT)

Video encoder

Fusion via cross-attention or transformer layers

Classification into 4 answer choices

Self-Supervised Improvement (optional)

Generate pseudo labels on unlabeled public test set

Retrain model using mixture of hard labels + pseudo labels

Ensemble

Combine multiple improved checkpoints to boost accuracy

📈 Results
| Model Version                  | Accuracy   |
| ------------------------------ | ---------- |
| Baseline                       | ~0.50–0.52 |
| Improved Model + Self Training | ~0.58+     |
| Public Test | ~0.48+     |
