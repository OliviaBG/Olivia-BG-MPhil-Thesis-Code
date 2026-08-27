#!/usr/bin/env python3
"""
esm_finetune_kfold.py
============================================================
Proper 5-fold CV fine-tuning of ESM2-150M (esm2_t30_150M_UR50D) for NES
and/or NLS -- generalizes nls_esm_finetune.py's single 80/20-split version
to the same 5-fold protocol used by esm_full_comparison.py, so all the
numbers from both scripts are directly comparable for a single thesis
figure/table.

For each of the 5 outer folds, the outer training portion is further split
85/15 (inner) purely for early-stopping decisions -- the outer test fold is
NEVER used for early stopping, only for the fold's final reported metric,
so there's no leakage into the number that matters. A fresh copy of the
pretrained model is loaded every fold (first fold downloads the weights
once; every later fold reuses the local HuggingFace cache, no repeated
network calls).

Same partial fine-tuning approach as nls_esm_finetune.py and for the same
reason (760-1200 examples is too little to safely update all 150M params):
classifier head + last N transformer layers unfrozen (--unfreeze-layers,
default 2), class-weighted loss, differential learning rates (small for the
unfrozen ESM2 layers, larger for the fresh head).

Checkpointed per fold so it's safe to interrupt and rerun.

Needs a GPU (5 folds x 2 targets on CPU would be painfully slow).
Requires: pip install torch transformers scikit-learn numpy

Usage:
    python3 esm_finetune_kfold.py                  # both nes and nls
    python3 esm_finetune_kfold.py --target nls
    python3 esm_finetune_kfold.py --unfreeze-layers 4 --epochs 20
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def _load_data(target):
    if target == "nls":
        from nls_ml_predictor import NLSPredictor
        p = NLSPredictor()
    else:
        from nes_ml_predictor_improved import ImprovedNESPredictor
        p = ImprovedNESPredictor()
    dataset = p.build_training_dataset()
    seqs = [d["seq"].upper() for d in dataset["positives"]] + [d["seq"].upper() for d in dataset["negatives"]]
    labels = [1] * len(dataset["positives"]) + [0] * len(dataset["negatives"])
    print(f"  {target}: {len(seqs)} sequences ({sum(labels)} positive / {len(labels) - sum(labels)} negative)")
    return seqs, labels


def _unfreeze(model, n_layers, num_hidden_layers):
    unfrozen_idxs = set(range(num_hidden_layers - n_layers, num_hidden_layers))
    n_trainable = 0
    for name, param in model.named_parameters():
        unfreeze = "classifier" in name
        if not unfreeze:
            m = re.search(r"encoder\.layer\.(\d+)\.", name)
            unfreeze = bool(m and int(m.group(1)) in unfrozen_idxs)
        param.requires_grad = unfreeze
        if unfreeze:
            n_trainable += param.numel()
    return n_trainable


def _train_one_fold(seq_train, y_train, seq_test, y_test, tokenizer, args, device, fold_seed):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForSequenceClassification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

    seq_tr, seq_val, y_tr, y_val = train_test_split(
        seq_train, y_train, test_size=0.15, random_state=fold_seed, stratify=y_train)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    for p in model.parameters():
        p.requires_grad = False
    n_trainable = _unfreeze(model, args.unfreeze_layers, model.config.num_hidden_layers)

    class SeqDataset(Dataset):
        def __init__(self, seqs, labels):
            self.seqs, self.labels = seqs, labels
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, i):
            return self.seqs[i], self.labels[i]

    def collate(batch):
        b_seqs, b_labels = zip(*batch)
        enc = tokenizer(list(b_seqs), return_tensors="pt", padding=True, truncation=True, max_length=64)
        return enc, torch.tensor(b_labels, dtype=torch.long)

    train_loader = DataLoader(SeqDataset(seq_tr, y_tr), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SeqDataset(seq_val, y_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(SeqDataset(seq_test, y_test), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    n_pos, n_neg = sum(y_tr), len(y_tr) - sum(y_tr)
    class_weight = torch.tensor(
        [len(y_tr) / (2 * max(n_neg, 1)), len(y_tr) / (2 * max(n_pos, 1))], dtype=torch.float).to(device)
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

    best_val_f1, best_state, epochs_no_improve = -1.0, None, 0
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

        val_metrics = run_eval(val_loader)
        print(f"    epoch {epoch:2d}  train_loss={total_loss / len(seq_tr):.4f}  "
              f"val_F1={val_metrics['f1']:.3f}  val_AUC={val_metrics['auc']:.3f}")
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"    early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_metrics = run_eval(test_loader)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return test_metrics, n_trainable


def run_target(target, args, out_dir):
    import torch
    from transformers import AutoTokenizer
    from sklearn.model_selection import StratifiedKFold

    print(f"\n{'=' * 60}\n{target.upper()} -- fine-tune, 5-fold CV (max {args.epochs} epochs/fold)\n{'=' * 60}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU found -- will be slow)"))

    seqs, labels = _load_data(target)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ckpt_path = out_dir / f"esm_finetune_kfold_{target}.json"
    results = json.load(open(ckpt_path)) if ckpt_path.exists() else {"target": target, "folds": []}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    seqs_arr, labels_arr = np.array(seqs, dtype=object), np.array(labels)

    done_folds = len(results["folds"])
    for i, (tr_idx, te_idx) in enumerate(cv.split(seqs_arr, labels_arr)):
        if i < done_folds:
            continue
        print(f"\n  Fold {i + 1}/5...")
        t0 = time.time()
        seq_train, y_train = seqs_arr[tr_idx].tolist(), labels_arr[tr_idx].tolist()
        seq_test, y_test = seqs_arr[te_idx].tolist(), labels_arr[te_idx].tolist()
        test_metrics, n_trainable = _train_one_fold(
            seq_train, y_train, seq_test, y_test, tokenizer, args, device, fold_seed=42 + i)
        print(f"  Fold {i + 1} test: F1={test_metrics['f1']:.3f}  AUC={test_metrics['auc']:.3f}  "
              f"P={test_metrics['precision']:.3f}  R={test_metrics['recall']:.3f}  ({time.time() - t0:.0f}s)")
        results["folds"].append(test_metrics)
        results["n_trainable_params"] = n_trainable
        json.dump(results, open(ckpt_path, "w"), indent=2)

    f1s = [f["f1"] for f in results["folds"]]
    aucs = [f["auc"] for f in results["folds"]]
    results["f1_mean"], results["f1_std"] = float(np.mean(f1s)), float(np.std(f1s))
    results["auc_mean"], results["auc_std"] = float(np.mean(aucs)), float(np.std(aucs))
    json.dump(results, open(ckpt_path, "w"), indent=2)
    print(f"\n{target.upper()} fine-tuned 5-fold CV: F1={results['f1_mean']:.3f}+/-{results['f1_std']:.3f}  "
          f"AUC={results['auc_mean']:.3f}+/-{results['auc_std']:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["nes", "nls"], default=None, help="omit to run both")
    ap.add_argument("--unfreeze-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-esm", type=float, default=2e-5)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out_dir = Path(args.out)
    targets = [args.target] if args.target else ["nls", "nes"]
    for target in targets:
        run_target(target, args, out_dir)


if __name__ == "__main__":
    main()
