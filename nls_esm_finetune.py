#!/usr/bin/env python3
"""
nls_esm_finetune.py
============================================================
Fine-tunes ESM2-150M (esm2_t30_150M_UR50D) on the NLS positive/negative
training pool, instead of using it as a frozen feature extractor
(esm_embed_nls_only.py / nls_esm_pca_test.py already showed frozen
embeddings, with or without PCA, don't beat the hand-engineered feature
set -- this tests whether letting the language model's own weights adapt
to the task does any better).

Approach (standard small-dataset fine-tuning practice, not full fine-tuning
-- ~760 examples is too little to safely update all 150M params without
just memorizing the training set):
  - AutoModelForSequenceClassification (adds a classification head on top
    of ESM2 automatically -- no separate pooling code needed).
  - Freeze everything except the classification head + the last N
    transformer layers (--unfreeze-layers, default 2). This is the usual
    compromise: let the top of the network specialize for NLS-vs-not,
    keep the lower layers' general protein-sequence knowledge intact.
  - Differential learning rates: small LR for the unfrozen ESM2 layers,
    larger LR for the freshly-initialized head.
  - Class-weighted loss (same 'balanced' philosophy as every classifier
    in nls_ml_predictor.py) since positives/negatives aren't 50/50.
  - Single stratified 80/20 train/val split (not 5-fold CV like the
    hand-engineered/frozen-embedding comparisons) -- 5x the fine-tuning
    cost for 5-fold CV isn't worth it for a first feasibility check, so
    this number isn't directly apples-to-apples with the 0.879 F1
    hand-engineered baseline, just a first read on whether fine-tuning
    is worth pursuing further.
  - Early stopping on validation F1.

This is a standalone diagnostic script -- it does NOT modify
nls_ml_predictor.py or its shipped model. Run it, look at the result,
then decide whether it's worth building into the real pipeline.

Needs a GPU. Requires: pip install torch transformers scikit-learn numpy

Usage:
    python3 nls_esm_finetune.py
    python3 nls_esm_finetune.py --unfreeze-layers 4 --epochs 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
THIS_DIR = Path(__file__).resolve().parent


def _load_data():
    sys.path.insert(0, str(THIS_DIR))
    from nls_ml_predictor import NLSPredictor

    print("Loading NLS training set...")
    predictor = NLSPredictor()
    dataset = predictor.build_training_dataset()
    seqs = [p["seq"].upper() for p in dataset["positives"]] + [n["seq"].upper() for n in dataset["negatives"]]
    labels = [1] * len(dataset["positives"]) + [0] * len(dataset["negatives"])
    # de-dup by sequence (same convention as build_training_dataset/embedding scripts)
    seen = {}
    for s, y in zip(seqs, labels):
        seen[s] = y  # last-wins on duplicate seq, matches dataset's own dedup behavior
    seqs = list(seen.keys())
    labels = list(seen.values())
    print(f"  {len(seqs)} unique sequences ({sum(labels)} positive / {len(labels) - sum(labels)} negative)")
    return seqs, labels


def _unfreeze(model, n_layers, num_hidden_layers):
    import re
    unfrozen_layer_idxs = set(range(num_hidden_layers - n_layers, num_hidden_layers))
    n_unfrozen_params = 0
    for name, param in model.named_parameters():
        unfreeze = False
        if "classifier" in name:
            unfreeze = True
        else:
            m = re.search(r"encoder\.layer\.(\d+)\.", name)
            if m and int(m.group(1)) in unfrozen_layer_idxs:
                unfreeze = True
        param.requires_grad = unfreeze
        if unfreeze:
            n_unfrozen_params += param.numel()
    print(f"  Unfrozen: classifier head + last {n_layers} of {num_hidden_layers} transformer layers "
          f"({n_unfrozen_params:,} trainable params)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unfreeze-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-esm", type=float, default=2e-5)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="esm_finetune_results.json")
    ap.add_argument("--save-model", action="store_true",
                     help="also save the fine-tuned weights to esm_finetuned_model/ (~600MB)")
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU found -- will be slow)"))

    seqs, labels = _load_data()
    seq_train, seq_val, y_train, y_val = train_test_split(
        seqs, labels, test_size=args.val_size, random_state=args.seed, stratify=labels)
    print(f"  train={len(seq_train)} ({sum(y_train)} pos)  val={len(seq_val)} ({sum(y_val)} pos)")

    print(f"\nLoading {MODEL_NAME} with a classification head...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    num_hidden_layers = model.config.num_hidden_layers
    for p in model.parameters():
        p.requires_grad = False
    _unfreeze(model, args.unfreeze_layers, num_hidden_layers)

    class SeqDataset(Dataset):
        def __init__(self, seqs, labels):
            self.seqs, self.labels = seqs, labels
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return self.seqs[idx], self.labels[idx]

    def collate(batch):
        b_seqs, b_labels = zip(*batch)
        enc = tokenizer(list(b_seqs), return_tensors="pt", padding=True, truncation=True, max_length=64)
        return enc, torch.tensor(b_labels, dtype=torch.long)

    train_loader = DataLoader(SeqDataset(seq_train, y_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SeqDataset(seq_val, y_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    n_pos, n_neg = sum(y_train), len(y_train) - sum(y_train)
    class_weight = torch.tensor([len(y_train) / (2 * n_neg), len(y_train) / (2 * n_pos)], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n]
    esm_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
    optimizer = torch.optim.AdamW([
        {"params": esm_params, "lr": args.lr_esm},
        {"params": head_params, "lr": args.lr_head},
    ])

    def run_eval(loader):
        model.eval()
        all_proba, all_labels = [], []
        with torch.no_grad():
            for enc, y in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_proba.extend(proba.tolist())
                all_labels.extend(y.numpy().tolist())
        pred = [1 if p >= 0.5 else 0 for p in all_proba]
        return {
            "f1": f1_score(all_labels, pred, zero_division=0),
            "auc": roc_auc_score(all_labels, all_proba) if len(set(all_labels)) > 1 else float("nan"),
            "precision": precision_score(all_labels, pred, zero_division=0),
            "recall": recall_score(all_labels, pred, zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, pred).tolist(),
        }

    print(f"\nFine-tuning ({args.epochs} max epochs, patience {args.patience}, "
          f"lr_esm={args.lr_esm}, lr_head={args.lr_head}, batch_size={args.batch_size})...")
    best_val_f1, best_epoch, best_metrics, epochs_no_improve = -1.0, -1, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for enc, y in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(**enc).logits
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
        train_loss = total_loss / len(seq_train)

        val_metrics = run_eval(val_loader)
        print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  "
              f"val_F1={val_metrics['f1']:.3f}  val_AUC={val_metrics['auc']:.3f}  "
              f"val_P={val_metrics['precision']:.3f}  val_R={val_metrics['recall']:.3f}  "
              f"elapsed={time.time()-t0:.0f}s")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1, best_epoch, best_metrics = val_metrics["f1"], epoch, val_metrics
            epochs_no_improve = 0
            if args.save_model:
                model.save_pretrained(THIS_DIR / "esm_finetuned_model")
                tokenizer.save_pretrained(THIS_DIR / "esm_finetuned_model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping (no val F1 improvement in {args.patience} epochs)")
                break

    print(f"\nBest epoch: {best_epoch}  val_F1={best_metrics['f1']:.3f}  val_AUC={best_metrics['auc']:.3f}")
    print(f"  precision={best_metrics['precision']:.3f}  recall={best_metrics['recall']:.3f}")
    print(f"  confusion matrix [[TN,FP],[FN,TP]]: {best_metrics['confusion_matrix']}")

    result = {
        "model": MODEL_NAME, "unfreeze_layers": args.unfreeze_layers,
        "epochs_run": epoch, "best_epoch": best_epoch,
        "lr_esm": args.lr_esm, "lr_head": args.lr_head, "batch_size": args.batch_size,
        "n_train": len(seq_train), "n_val": len(seq_val),
        "best_val_metrics": best_metrics,
        "note": ("Single 80/20 stratified split, not 5-fold CV -- not directly "
                 "comparable to the 0.879 F1 5-fold hand-engineered baseline, "
                 "this is a first feasibility read."),
    }
    with open(THIS_DIR / args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {args.out}")
    if args.save_model:
        print(f"Saved fine-tuned model to esm_finetuned_model/")


if __name__ == "__main__":
    main()
