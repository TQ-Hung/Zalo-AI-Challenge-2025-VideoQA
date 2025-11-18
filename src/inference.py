import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.datasets import ZaloDataset
from src.model import VideoQAModel


# ============================================================
#                 INFERENCE (FULL FIXED VERSION)
# ============================================================
def inference(
    model_path: str,
    data_path: str,
    feature_dir: str,
    output_path: str,
    model_name: str = "vinai/phobert-base-v2",
    batch_size: int = 8,
    max_length: int = 128,
    device: str = None,
):
    """
    - model_path: checkpoint .pt
    - data_path: test json file
    - feature_dir: extracted video features
    - output_path: output submission.jsonl
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------
    # Load tokenizer & dataset
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_ds = ZaloDataset(
        data_path,
        tokenizer,
        feature_dir,
        split="test",
        max_length=max_length,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=test_ds.collate_fn,
    )

    # ------------------------
    # Load model
    # ------------------------
    print(f"Loading model from {model_path} ...")

    checkpoint = torch.load(model_path, map_location=device)
    model = VideoQAModel()
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    all_preds = []

    # Flags to print debug only once
    shape_reported = False
    sample_reported = False

    # ------------------------
    # Inference loop
    # ------------------------
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencing"):
            # Move batch to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # Forward pass
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["appearance_feats"],
                batch["motion_feats"],
            )

            # -----------------------------
            # DEBUG SHAPES (once)
            # -----------------------------
            if not shape_reported:
                print("DEBUG >>> logits.shape:", logits.shape)
                print("DEBUG >>> logits.dtype:", logits.dtype)
                shape_reported = True

            # -----------------------------
            # FIX CHOICE SCORING
            # logits expected shapes:
            #   (B, C, num_classes) OR (B, C) OR (B, C, 1)
            # -----------------------------
            if logits.dim() == 3:
                # Example: (B, C, num_classes)
                # Take max across class dimension → 1 score/choice
                choice_scores = logits.max(dim=2).values  # (B, C)

            elif logits.dim() == 2:
                # Already (B, C)
                choice_scores = logits

            elif logits.dim() == 4 and logits.size(-1) == 1:
                # Shape (B, C, 1) → squeeze
                choice_scores = logits.squeeze(-1)

            else:
                # Fallback: flatten last dims
                B = logits.size(0)
                C = logits.size(1)
                choice_scores = logits.view(B, C, -1).mean(dim=2)

            # -----------------------------
            # DEBUG SAMPLE (once)
            # -----------------------------
            if not sample_reported:
                print("DEBUG >>> sample choice_scores[0]:", choice_scores[0].cpu().numpy())
                sample_reported = True

            # -----------------------------
            # Final predicted choice index
            # -----------------------------
            preds_batch = choice_scores.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds_batch)

    # -----------------------------
    # Convert to labels A/B/C/D
    # -----------------------------
    id_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    final_labels = [id_to_label[int(x)] for x in all_preds]

    # -----------------------------
    # Save output
    # -----------------------------
    print("Saving output to:", output_path)

    with open(data_path, "r", encoding="utf-8") as f:
        raw = [json.loads(x) for x in f]

    assert len(raw) == len(final_labels)

    with open(output_path, "w", encoding="utf-8") as g:
        for item, label in zip(raw, final_labels):
            out = {"id": item["id"], "answer": label}
            g.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("DONE.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)

    args = parser.parse_args()

    inference(
        model_path=args.model_path,
        data_path=args.data_path,
        feature_dir=args.feature_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
