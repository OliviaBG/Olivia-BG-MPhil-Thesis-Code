# CRM1 reference structures

Cleaned reference structures for the CRM1 (exportin-1 / XPO1) NES groove,
used by `pocket_detector.py` for pocket compatibility and by
`md_refinement.py` as the docking target.

| Pattern | Contents |
| --- | --- |
| `CRM1_Ran_<PDB>.pdb` | chains A + C only (CRM1 + RanGTP), cargo and duplicate chains removed |
| `CRM1_Ran_<PDB>_v2clean.pdb` | same, rebuilt by `build_clean_crystal_references.py` with the asymmetric-unit multi-copy handling corrected |
| `NES_peptide_<PDB>_chain<X>.pdb` | the crystallographic NES peptide extracted from that entry |
| `PKI_NES_peptide_3NBY_chainB_4-13.pdb` | the PKI NES, used as the positive control for docking |
| `CRM1_Ran_only.pdb` | the default reference `app.py` loads |
| `pocket_analysis.json` | cached pocket geometry for the default reference |

Source entries: **3NBY, 3NBZ, 3NC0, 3GB8, 3GJX, 5DHF, 5DIF, 5UWH, 5UWS,
5UWU**, all from the RCSB PDB and subject to its terms of use.

The raw `*_original.pdb` downloads are not committed. Regenerate the whole
directory from scratch with:

```bash
python setup_crm1_reference.py            # fetch from RCSB
python extract_crystal_references.py      # extract CRM1+Ran and the NES peptides
python build_clean_crystal_references.py  # rebuild the v2clean set
```

Why the cargo chains are stripped: the raw crystal structures contain a
bound cargo protein occupying the NES groove, so pocket detection run on the
unmodified file finds the groove already filled and scores every candidate
as incompatible.

Groove-shell caches (`*_groove_shell_cache.json`) are regenerated
automatically by `md_refinement.py` and are not committed.
