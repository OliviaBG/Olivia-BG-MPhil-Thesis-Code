#!/usr/bin/env python3
"""
esm_embed_nls_only.py
============================================================
Standalone, NLS-only version of esm_embed_sequences.py -- computes frozen
ESM2-150M (esm2_t30_150M_UR50D) per-sequence embeddings for every unique NLS
training sequence and caches them to disk, matching the exact convention
already used in esm_embeddings/nls_embeddings.npz (mean-pooled over
non-padding tokens, last hidden layer) so the output merges cleanly with
that existing cache.

Only needs nls_ml_predictor.py + nls_data_pipeline/nls_dataset.csv +
nls_data_pipeline/nls_negatives.csv -- no NES files required (unlike the
original combined script).

Run on a machine with network access to huggingface.co and (ideally) a GPU:
    pip install torch transformers numpy
    python3 esm_embed_nls_only.py

Output: esm_embeddings/nls_embeddings.npz  ({sequences, vectors}, 640-dim)
"""
import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
THIS_DIR = Path(__file__).resolve().parent


def _collect_sequences():
    import sys
    sys.path.insert(0, str(THIS_DIR))
    from nls_ml_predictor import NLSPredictor

    print("Loading NLS training set (for its sequence list only)...")
    predictor = NLSPredictor()
    dataset = predictor.build_training_dataset()
    seqs = sorted({p["seq"].upper() for p in dataset["positives"]} |
                  {n["seq"].upper() for n in dataset["negatives"]})
    print(f"  {len(seqs)} unique NLS sequences")
    return seqs


def _embed_batch(sequences, tokenizer, model, device, batch_size):
    import torch

    all_vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            all_vecs.append(mean_pooled)
            done = min(i + batch_size, len(sequences))
            print(f"  {done}/{len(sequences)} embedded", end="\r")
    print()
    return np.concatenate(all_vecs, axis=0)


def main():
    out_dir = THIS_DIR / "esm_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU found)"))

    print(f"\nLoading {MODEL_NAME} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    hidden_dim = model.config.hidden_size
    print(f"  Loaded in {time.time() - t0:.1f}s -- hidden dim {hidden_dim}")

    seqs = _collect_sequences()

    print(f"\nEmbedding {len(seqs)} NLS sequences (batch size 64) ...")
    t0 = time.time()
    vecs = _embed_batch(seqs, tokenizer, model, device, batch_size=64)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / max(1, len(seqs)):.3f}s/seq)")

    np.savez_compressed(out_dir / "nls_embeddings.npz", sequences=np.array(seqs), vectors=vecs)

    manifest = {
        "model": MODEL_NAME, "hidden_dim": int(hidden_dim), "device": device,
        "pooling": "mean over non-padding tokens, last hidden layer",
        "n_nls_sequences": len(seqs), "nls_embed_seconds": round(elapsed, 1),
    }
    with open(out_dir / "embedding_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {out_dir}/nls_embeddings.npz ({hidden_dim}-dim vectors, {MODEL_NAME})")


if __name__ == "__main__":
    main()
