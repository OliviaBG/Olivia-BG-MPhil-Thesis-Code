#!/usr/bin/env python3
"""
esm_embed_sequences.py
============================================================
Computes FROZEN ESM2-150M (esm2_t30_150M_UR50D) per-sequence embeddings
for every unique NES and NLS positive/negative training sequence, and
caches them to disk. This is the first step for esm_model_comparison.py
and esm_umap_agreement.py.

"Frozen" = the pretrained language model's weights are never updated --
this only runs it forward to get a fixed-size vector representation per
sequence (mean-pooled over the per-residue token embeddings from its last
hidden layer), the same convention used in the ESM/NLSExplorer literature.
No fine-tuning happens here.

Needs a GPU for reasonable speed (falls back to CPU automatically if none
found, just slower). Requires: pip install torch transformers

Usage:
    python3 esm_embed_sequences.py
    python3 esm_embed_sequences.py --batch-size 64 --out esm_embeddings

Outputs (written to --out, default 'esm_embeddings/'):
    nes_embeddings.npz   {sequence: 150M-model hidden-dim vector} for every
                          unique NES positive/negative sequence
    nls_embeddings.npz   same, for NLS
    embedding_manifest.json   model name, dim, n sequences, run metadata
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = 'facebook/esm2_t30_150M_UR50D'
THIS_DIR = Path(__file__).resolve().parent


def _collect_sequences():
    """Reuses each predictor's own build_training_dataset() so the sequence
    set is guaranteed identical to what actually gets trained on -- no
    separately-maintained copy that could drift."""
    import sys
    sys.path.insert(0, str(THIS_DIR))

    from nes_ml_predictor_improved import ImprovedNESPredictor
    from nls_ml_predictor import NLSPredictor

    print("Loading NES training set (for its sequence list only)...")
    nes_predictor = ImprovedNESPredictor()
    nes_dataset = nes_predictor.build_training_dataset()
    nes_seqs = sorted({p['seq'].upper() for p in nes_dataset['positives']} |
                       {n['seq'].upper() for n in nes_dataset['negatives']})
    print(f"  {len(nes_seqs)} unique NES sequences")

    print("Loading NLS training set (for its sequence list only)...")
    nls_predictor = NLSPredictor()
    nls_dataset = nls_predictor.build_training_dataset()
    nls_seqs = sorted({p['seq'].upper() for p in nls_dataset['positives']} |
                       {n['seq'].upper() for n in nls_dataset['negatives']})
    print(f"  {len(nls_seqs)} unique NLS sequences")

    return nes_seqs, nls_seqs


def _embed_batch(sequences, tokenizer, model, device, batch_size):
    """Mean-pool the last hidden layer over real (non-padding) tokens for
    each sequence -- the standard frozen-ESM-embedding convention."""
    import torch

    all_vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                             max_length=64)  # NES/NLS windows are short (<=25 aa); generous ceiling
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state  # (batch, seq_len, dim)
            mask = enc['attention_mask'].unsqueeze(-1).float()  # (batch, seq_len, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            all_vecs.append(mean_pooled)
            done = min(i + batch_size, len(sequences))
            print(f"  {done}/{len(sequences)} embedded", end='\r')
    print()
    return np.concatenate(all_vecs, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--out', default='esm_embeddings')
    ap.add_argument('--model', default=MODEL_NAME)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else " (no GPU found -- will be slow)"))

    print(f"\nLoading {args.model} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    hidden_dim = model.config.hidden_size
    print(f"  Loaded in {time.time() - t0:.1f}s -- hidden dim {hidden_dim}")

    nes_seqs, nls_seqs = _collect_sequences()

    print(f"\nEmbedding {len(nes_seqs)} NES sequences (batch size {args.batch_size}) ...")
    t0 = time.time()
    nes_vecs = _embed_batch(nes_seqs, tokenizer, model, device, args.batch_size)
    nes_elapsed = time.time() - t0
    print(f"  Done in {nes_elapsed:.1f}s ({nes_elapsed / max(1, len(nes_seqs)):.3f}s/seq)")

    print(f"\nEmbedding {len(nls_seqs)} NLS sequences (batch size {args.batch_size}) ...")
    t0 = time.time()
    nls_vecs = _embed_batch(nls_seqs, tokenizer, model, device, args.batch_size)
    nls_elapsed = time.time() - t0
    print(f"  Done in {nls_elapsed:.1f}s ({nls_elapsed / max(1, len(nls_seqs)):.3f}s/seq)")

    np.savez_compressed(out_dir / 'nes_embeddings.npz',
                         sequences=np.array(nes_seqs), vectors=nes_vecs)
    np.savez_compressed(out_dir / 'nls_embeddings.npz',
                         sequences=np.array(nls_seqs), vectors=nls_vecs)

    manifest = {
        'model': args.model, 'hidden_dim': int(hidden_dim), 'device': device,
        'pooling': 'mean over non-padding tokens, last hidden layer',
        'n_nes_sequences': len(nes_seqs), 'n_nls_sequences': len(nls_seqs),
        'nes_embed_seconds': round(nes_elapsed, 1), 'nls_embed_seconds': round(nls_elapsed, 1),
    }
    with open(out_dir / 'embedding_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {out_dir}/nes_embeddings.npz and {out_dir}/nls_embeddings.npz")
    print(f"  ({hidden_dim}-dim vectors, {args.model})")


if __name__ == '__main__':
    main()
