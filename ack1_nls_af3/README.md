# ACK NLS: register and presentation analysis

The part of the ACK NLS argument that can be made without a force field.
These scripts ask whether the K71-R72-K73 basic cluster can satisfy the same
importin-alpha pockets a canonical classical NLS satisfies, and what
presenting it would cost the folded SAM domain.

| Script | Question it answers |
| --- | --- |
| `thread_ack1.py` | Mapped onto the crystallographic cNLS register, which anchor pockets does ACK satisfy and which does it miss? |
| `template_geometry.py` | What backbone geometry does the bound peptide adopt in the template, and can ACK reach it? |
| `helix_cost.py` | What does it cost to unwind the SAM alpha-5 helix enough to expose the cluster? |
| `presentation_cost.py` | Combined free-energy cost of presenting the signal from the folded domain |
| `dimer_importin_clash.py` | Is the site sterically occluded in the SAM dimer? |
| `figures_ack1_nls.py`, `figures_ack1_receptors.py` | Figures, from the outputs of the above and of `../ack1_importin_gpu/` |

Templates used: **3UL1** chain A (nucleoplasmin 152-172, bipartite, occupying
both the minor and major sites) and **1EJL** chains A and B (SV40 large T
126-132, minor and major site respectively).

`af3_jobs.json` and `af3_helix_jobs.json` are the AlphaFold 3 job
definitions used for the structure-prediction cross-check.

## Required input

`thread_ack1.py` and `template_geometry.py` read `ack1.fasta` from the
working directory, and `thread_ack1.py` reads `3UL1.pdb` and `1EJL.pdb` from
the repository root (both committed).

```bash
curl -o ack1.fasta https://rest.uniprot.org/uniprotkb/Q07912.fasta
```

This analysis is deliberately **not** an energy calculation. It reports
which pockets are chemically satisfied, which are not, and where ACK
departs from a canonical cNLS. The energetics live in
`../ack1_importin_gpu/`.
