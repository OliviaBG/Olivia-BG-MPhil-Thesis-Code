"""
Enhanced Pocket Detector with CRM1 Structure Knowledge
Uses the CRM1-NES crystal structure (3NBY) as a reference template
"""

import subprocess
import tempfile
import shutil
import time
from collections import Counter
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from io import StringIO


# CRM1 NES-binding pocket characteristics from crystal structure (PDB: 3NBY)
# Extracted from the actual CRM1-PKI NES complex.
#
# The previous 14-residue list here was WRONG -- verified by direct
# 5A-distance computation against the real grafted PKI NES peptide in the
# pristine crm1_reference/3NBY_original.pdb (chain A = CRM1, confirmed against
# RCSB's own 3NBY entry). Only 5-6 of those 14 residues actually contact the
# NES; the rest (e.g. PHE366/369/414, MET425/428, VAL415, LEU411, ILE489,
# LEU568) are not real contacts in this structure. This was a hardcoded-data
# bug in this Python list, independent of which reference PDB file gets
# loaded (identical mismatches reproduced across the original RCSB download,
# the chain-filtered CRM1_Ran_only.pdb, and the PDBFixer-repaired version).
#
# Replaced with the real, verified 13-residue contact set (chain A residues
# within 5A of the bound NES peptide):
CRM1_BINDING_POCKET = {
    'center_residues': [
        # Core hydrophobic groove residues (verified within 5Å of the real
        # grafted PKI NES peptide in 3NBY, chain A -- see comment above)
        'VAL518', 'ILE521', 'LEU525', 'CYS528', 'ALA538', 'ALA541',
        'ILE544', 'MET545', 'PHE554', 'PHE561', 'VAL565', 'PHE572', 'VAL580'
    ],

    # Recomputed from the real CA coordinates of the 13 verified
    # residues above, in the production reference file
    # (crm1_reference/CRM1_Ran_only.pdb, chains A+C only).
    'pocket_center_coords': (59.0, -71.4, -13.0),

    'pocket_dimensions': {
        'length': 23.0,  # Å
        'width': 14.6,   # Å
        'depth': 8.1     # Å
    },

    'hydrophobicity_profile': {
        'Φ0': ['L', 'I', 'V', 'M', 'F'],  # Hydrophobic pocket 0
        'Φ1': ['L', 'I', 'V', 'M'],        # Hydrophobic pocket 1
        'Φ2': ['L', 'I', 'V', 'F'],        # Hydrophobic pocket 2
        'Φ3': ['L', 'I', 'V', 'F'],        # Hydrophobic pocket 3
        'Φ4': ['L', 'I', 'V', 'M']         # Hydrophobic pocket 4
    },

    'groove_shape': 'extended_hydrophobic_cleft',
    'required_features': [
        'hydrophobic_surface',
        'elongated_groove',
        'surface_exposed',
        'low_charge_density'
    ]
}

# Real residue-type profile of CRM1's actual 14-residue contact
# surface (the same 'center_residues' list above, from 3NBY, <5A of the
# bound NES), used by _composition_similarity() below to compare a
# candidate pocket's residue-type mix against CRM1's real groove instead of
# one generic averaged hydrophobicity number. Built directly from the
# 3-letter code prefix of each entry -- always available even when no
# CRM1 reference PDB is loaded (unlike self.crm1_pocket_template, which
# needs a real structure file).
_CRM1_POCKET_RESNAME_COUNTS = Counter(res[:3] for res in CRM1_BINDING_POCKET['center_residues'])
CRM1_POCKET_COMPOSITION = {
    aa: count / len(CRM1_BINDING_POCKET['center_residues'])
    for aa, count in _CRM1_POCKET_RESNAME_COUNTS.items()
}  # {'PHE': 0.286, 'MET': 0.214, 'LEU': 0.214, 'ILE': 0.143, 'VAL': 0.071, 'CYS': 0.071}

# CRM1's real NES-binding groove is almost entirely hydrophobic (none of
# the 14 contact residues above are charged) -- used by the charge-density
# check, which was previously unimplemented (see
# _filter_for_crm1_compatibility's old step 5).
CHARGED_RESNAMES = {'ASP', 'GLU', 'LYS', 'ARG'}

# Standard hydrophobic residue set (3-letter codes) -- used to compute a
# real, bounded 0-1 hydrophobic-residue fraction for a candidate pocket's
# own lining residues, replacing reliance on real fpocket's raw
# "Hydrophobicity score" output field (an unnormalized internal metric,
# NOT a 0-1 ratio -- see the fix in _filter_for_crm1_compatibility
# for the full story of the bug this caused).
HYDROPHOBIC_RESNAMES = {'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'MET', 'TRP', 'CYS'}

# Average side-chain-inclusive residue volumes (Å³), Zamyatnin 1972 (Prog
# Biophys Mol Biol 24:107-123). Used by _residue_residue_match to compare
# individual candidate-pocket residues against individual CRM1 contact
# residues by real side-chain size, instead of only the aggregate
# bag-of-residues composition check above -- this is what actually lets
# Güttler et al. 2010's "bulky at some positions, small at others" pattern
# (see CRM1_BINDING_POCKET's real residues: ALA538/ALA541 are small,
# PHE554/PHE561/PHE572 are bulky) be checked position-by-position.
RESIDUE_VOLUME = {
    'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
    'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
    'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
    'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0,
}


class CRM1AwarePocketDetector:
    """
    Enhanced pocket detector with CRM1-specific knowledge
    Uses both fpocket and structure-based template matching
    """

    def __init__(self, crm1_reference_path=None, fpocket_timeout=60, pocket_filter_timeout=90):
        self.fpocket_path = self._find_fpocket()
        self.crm1_reference_path = crm1_reference_path
        # Fpocket runtime scales with structure size and a fixed
        # 60s cap silently kills it on large proteins (e.g. O15078, ~2480
        # residues, timed out during evaluate_crm1_pocket_signal.py's real
        # run), forcing a fall-through to the weaker geometry detector for
        # no biological reason. This matters here specifically because the
        # negative pool is 90% coiled_coil, and coiled-coil-forming proteins
        # (cytoskeletal/motor/scaffold proteins) skew large -- so a fixed
        # timeout risks correlating detection_method with feature_kind
        # rather than with anything about NES-relevant pocket geometry.
        # Left at 60s by default (production app.py's live requests need to
        # stay responsive); offline batch tools like the evaluation script
        # can pass a larger value.
        self.fpocket_timeout = fpocket_timeout

        # This timeout only ever bounded the fpocket BINARY call
        # itself. _filter_for_crm1_compatibility, which runs after fpocket
        # returns and scores every detected pocket against CRM1's reference
        # groove residue-by-residue, had no timeout of its own -- found via
        # a real hang on Myosin-9 (P35579, 1960 residues, elongated
        # coiled-coil), which produces many candidate pockets along its
        # length for this uncapped per-pocket loop to work through. A
        # signal.alarm()-based timeout isn't safe here (Flask's threaded
        # dev server and this project's own test scripts call this from
        # non-main threads, where signal.alarm raises ValueError), so this
        # is enforced as a plain wall-clock check inside the loop itself
        # instead -- see _filter_for_crm1_compatibility. Returns whatever
        # pockets were already scored rather than none, same
        # graceful-degradation spirit as the fpocket subprocess timeout
        # falling back to geometry-based detection.
        self.pocket_filter_timeout = pocket_filter_timeout

        # Load CRM1 reference if provided
        # IMPORTANT: Reference should be CRM1 (chain A) + RanGTP (chain C) ONLY
        # The NES peptide (chain B) should be REMOVED so the binding cleft is exposed
        if crm1_reference_path and Path(crm1_reference_path).exists():
            self._load_crm1_reference()
        else:
            print("No CRM1 reference structure - using generic pocket detection")
            print("For best results, provide 3NBY with only chains A (CRM1) + C (Ran)")
            self.crm1_pocket_template = None

    def _find_fpocket(self):
        """Locate fpocket executable"""
        candidates = [
            'fpocket',
            '/usr/bin/fpocket',
            '/usr/local/bin/fpocket',
            '/opt/homebrew/bin/fpocket'
        ]

        for candidate in candidates:
            if shutil.which(candidate):
                print(f"Found fpocket at: {candidate}")
                return candidate

        print("Warning: fpocket not found - will use geometry-based fallback")
        return None

    def _load_crm1_reference(self):
        """
        Load and analyze CRM1 reference structure

        IMPORTANT: The reference PDB should contain:
        - Chain A: CRM1 (Exportin-1)
        - Chain C: RanGTP
        - NO Chain B (NES peptide should be removed to expose the cleft)

        This allows fpocket to detect the EMPTY binding groove that your
        target proteins will bind to.
        """
        try:
            parser = PDBParser(QUIET=True)
            crm1_structure = parser.get_structure('crm1', self.crm1_reference_path)

            # Extract the binding pocket region around Cys528
            # Since the NES is removed, we define the pocket by:
            # 1. Known binding residues from literature
            # 2. Hydrophobic surface patches
            # 3. Region around Cys528

            pocket_residues = []

            for model in crm1_structure:
                # Get chain A (CRM1)
                if 'A' not in model:
                    raise ValueError("Chain A (CRM1) not found in reference structure")

                chain_a = model['A']

                # Define the binding region around Cys528
                # Known binding residues from crystal structure analysis
                binding_region_residues = list(range(360, 580))  # Approximate binding region

                for residue in chain_a:
                    if residue.id[0] == ' ':  # Standard residue
                        res_num = residue.id[1]

                        # Include residues in binding region
                        if res_num in binding_region_residues:
                            # Focus on hydrophobic residues that form the groove
                            if residue.get_resname() in ['PHE', 'LEU', 'ILE', 'VAL',
                                                         'MET', 'CYS', 'ALA', 'TRP']:
                                pocket_residues.append(residue)

            # Calculate pocket signature from these residues
            pocket_coords = []
            for res in pocket_residues:
                for atom in res:
                    if atom.name == 'CA':
                        pocket_coords.append(atom.coord)

            if len(pocket_coords) == 0:
                raise ValueError("No pocket residues found")

            self.crm1_pocket_template = {
                'center': np.mean(pocket_coords, axis=0),
                'radius': np.max([np.linalg.norm(c - np.mean(pocket_coords, axis=0))
                                 for c in pocket_coords]),
                'residue_coords': pocket_coords,
                'num_residues': len(pocket_residues)
            }

            center = self.crm1_pocket_template['center']
            radius = self.crm1_pocket_template['radius']

            print(f"Loaded CRM1 reference pocket:")
            print(f"  - {len(pocket_residues)} hydrophobic residues")
            print(f"  - Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
            print(f"  - Radius: {radius:.1f} Å")
            print(f"  - This represents the EXPOSED binding cleft")

            # The block above pulls in EVERY hydrophobic residue
            # in the whole 360-580 span (a loose ~220-residue window, not
            # the real binding pocket) -- fine for a center/radius estimate,
            # too loose for genuine shape comparison. Separately extract the
            # real 14-residue contact geometry (CRM1_BINDING_POCKET's
            # center_residues, the actual <5A-of-NES residues from 3NBY) by
            # residue number, so _filter_for_crm1_compatibility can compare
            # a candidate pocket's shape against the REAL groove instead of
            # this looser blob.
            #
            # (same day, caught by inspection): matching by
            # residue NUMBER alone is not safe -- checked this project's own
            # crm1.pdb (generated by a third-party repair tool, "PRAS", per
            # its own header comment) and found 8 of the 14 expected contact
            # residues are a DIFFERENT amino acid at that residue number
            # than CRM1_BINDING_POCKET says (e.g. expected PHE414, this file
            # actually has SER414; expected MET425, this file has ALA425) --
            # no constant offset, just numbering that's drifted out of sync
            # with the original 3NBY numbering somewhere along this file's
            # processing history. Silently trusting residue number would
            # have quietly built tight_coords/tight_dims from a mix of 6
            # real contact residues and 8 wrong ones -- contaminated
            # geometry that LOOKS like it worked ("14/14 found") while
            # actually describing a different, partly-arbitrary region of
            # the protein. Now verifies the residue NAME at each number
            # matches what's expected before trusting the coordinate, and
            # reports exactly which ones didn't so a bad reference
            # structure is visible immediately instead of silently
            # poisoning every shape-match score downstream.
            expected_by_num = {
                int(''.join(filter(str.isdigit, r))): r[:3] for r in CRM1_BINDING_POCKET['center_residues']
            }
            tight_coords = []
            matched_residues = []
            mismatches = []
            for residue in chain_a:
                if residue.id[0] != ' ':
                    continue
                res_num = residue.id[1]
                if res_num not in expected_by_num or 'CA' not in residue:
                    continue
                expected_name = expected_by_num[res_num]
                actual_name = residue.get_resname()
                if actual_name == expected_name:
                    tight_coords.append(residue['CA'].coord)
                    matched_residues.append(f"{actual_name}{res_num}")
                else:
                    mismatches.append((res_num, expected_name, actual_name))

            if mismatches:
                print(f"  - WARNING: {len(mismatches)}/{len(expected_by_num)} expected CRM1 contact "
                      f"residues do NOT match this reference structure's actual residue at that "
                      f"number (numbering mismatch, not a real biological difference -- this "
                      f"reference file's chain A numbering has likely drifted out of sync with "
                      f"the original 3NBY numbering CRM1_BINDING_POCKET was built from):")
                for res_num, expected_name, actual_name in mismatches:
                    print(f"      expected {expected_name}{res_num}, found {actual_name}{res_num} instead")
                print(f"    -> Get a clean, unmodified 3NBY structure from RCSB PDB "
                      f"(https://www.rcsb.org/structure/3NBY) to fix this properly; verified "
                      f"only the {len(matched_residues)} name-matching residue(s) below are used.")

            if len(tight_coords) >= 3:
                tight_coords = np.array(tight_coords)
                self.crm1_pocket_template['tight_coords'] = tight_coords
                self.crm1_pocket_template['tight_dims'] = self._pocket_principal_dims(tight_coords)
                # Keep the matched residue NAMES alongside the
                # coordinates (same order) so _residue_residue_match can look
                # up each template residue's real side-chain volume, not
                # just its position.
                self.crm1_pocket_template['matched_residues'] = matched_residues
                print(f"  - Real contact geometry: {len(tight_coords)}/{len(expected_by_num)} "
                      f"VERIFIED (name-matched) residues used ({', '.join(matched_residues)}), "
                      f"dims (L/W/D) {[round(d, 1) for d in self.crm1_pocket_template['tight_dims']]} Å")
            else:
                self.crm1_pocket_template['tight_coords'] = None
                self.crm1_pocket_template['tight_dims'] = None
                self.crm1_pocket_template['matched_residues'] = None
                print(f"  - WARNING: only {len(tight_coords)} verified (name-matching) contact "
                      f"residues found in this reference structure -- too few for a shape "
                      f"comparison (need >=3). Shape comparison will fall back to the fpocket "
                      f"elongation flag instead until a correctly-numbered reference is provided.")

        except Exception as e:
            print(f"Warning: Could not load CRM1 reference: {e}")
            print(f"    Make sure the PDB has Chain A (CRM1) without Chain B (NES)")
            self.crm1_pocket_template = None

    def detect_pockets(self, pdb_content, residue_numbers=None, sequence=None):
        """
        Detect CRM1-compatible pockets with enhanced CRM1-specific validation

        Args:
            pdb_content: PDB file content as string
            residue_numbers: Optional list of residue numbers
            sequence: Optional protein sequence

        Returns:
            List of pockets with CRM1-compatibility scores
        """
        # Try fpocket first
        if self.fpocket_path:
            pockets = self._fpocket_detection(pdb_content)
        else:
            pockets = []

        # Fallback to geometry-based if needed
        # Tag every pocket with which detector actually produced
        # it. Previously there was no way to tell, from the output alone,
        # whether a given run used real fpocket (alpha-sphere geometry) or
        # the much weaker sliding-window fallback -- self.fpocket_path being
        # set only means the binary was found at startup, not that it
        # successfully found pockets for THIS structure (_fpocket_detection
        # silently returns [] on a non-zero exit code or exception, which
        # falls through to the fallback with no visible signal).
        if len(pockets) > 0:
            detection_method = 'fpocket'
        else:
            pockets = self._geometry_based_detection(pdb_content, residue_numbers)
            detection_method = 'geometry_fallback'

        for p in pockets:
            p['detection_method'] = detection_method

        # Apply CRM1-specific filtering and scoring
        crm1_compatible_pockets = self._filter_for_crm1_compatibility(
            pockets, pdb_content, sequence
        )

        return crm1_compatible_pockets

    def _fpocket_detection(self, pdb_content):
        """Run fpocket with CRM1-specific parameters"""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                pdb_file = tmpdir_path / 'structure.pdb'

                with open(pdb_file, 'w') as f:
                    f.write(pdb_content)

                # Run fpocket with parameters tuned for elongated grooves
                result = subprocess.run(
                    [self.fpocket_path, '-f', str(pdb_file),
                     '-m', '3.0',  # Min alpha sphere radius (smaller for grooves)
                     '-M', '6.0',  # Max alpha sphere radius
                     '-i', '30'],  # Min number of alpha spheres
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.fpocket_timeout
                )

                if result.returncode != 0:
                    print(f"Warning: fpocket exited with code {result.returncode} for this structure "
                          f"-- falling back to geometry-based detection. stderr: "
                          f"{result.stderr.strip()[:300] or '(none)'}")
                    return []

                pockets = self._parse_fpocket_output(tmpdir_path)
                if not pockets:
                    print("Warning: fpocket ran successfully but found 0 pockets for this structure "
                          "-- falling back to geometry-based detection.")
                return pockets

        except Exception as e:
            print(f"Warning: fpocket error: {e} -- falling back to geometry-based detection.")
            return []

    def _parse_fpocket_output(self, tmpdir):
        """
        Parse fpocket output files.

        NOTE (real bug found + fixed): this used to ONLY read
        structure_info.txt, which has summary stats per pocket (volume,
        druggability, hydrophobicity score, etc.) but no residue membership
        at all. Every pocket dict therefore never had a 'residue_numbers'
        key, so app.py's `pocket_dict` (which maps residue number -> pocket
        info for the "has this NES got a real pocket nearby" checks, the
        pocket_score contribution to combined_score, and the "CRM1 Pockets"
        3D color mode) was always empty regardless of how many real pockets
        fpocket found -- meaning every residue-indexed pocket lookup
        silently evaluated to 0/False for every protein, always. fpocket
        separately writes one PDB file per pocket
        (pockets/pocket{N}_atm.pdb) containing only the real protein atoms
        that line that specific pocket -- parsed below to get real residue
        numbers per pocket.
        """
        pockets = []
        output_dir = tmpdir / 'structure_out'

        if not output_dir.exists():
            return pockets

        info_file = output_dir / 'structure_info.txt'
        if info_file.exists():
            current_pocket = {}

            with open(info_file, 'r') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('Pocket'):
                        if current_pocket:
                            pockets.append(current_pocket)
                        current_pocket = {'id': len(pockets) + 1}

                    elif ':' in line and current_pocket:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().lower().replace(' ', '_').replace('-', '_')
                            value = parts[1].strip()

                            try:
                                if any(x in key for x in ['score', 'volume', 'druggability',
                                                          'hydrophobicity', 'density']):
                                    current_pocket[key] = float(value)
                                else:
                                    current_pocket[key] = value
                            except ValueError:
                                current_pocket[key] = value

            if current_pocket:
                pockets.append(current_pocket)

        # Real residue membership per pocket, from fpocket's own per-pocket
        # atom files -- info.txt numbers pockets 1-indexed ("Pocket 1",...)
        # but fpocket's pockets/ directory names files 0-indexed
        # (pocket0_atm.pdb,...); some fpocket builds instead use 1-indexed
        # filenames, so both are tried defensively.
        pockets_dir = output_dir / 'pockets'
        if pockets_dir.exists():
            for pocket in pockets:
                pocket_id = pocket.get('id', 0)
                candidate_files = [
                    pockets_dir / f'pocket{pocket_id - 1}_atm.pdb',
                    pockets_dir / f'pocket{pocket_id}_atm.pdb',
                ]
                atm_file = next((f for f in candidate_files if f.exists()), None)
                res_nums = set()
                if atm_file is not None:
                    try:
                        with open(atm_file) as f:
                            for line in f:
                                if line.startswith(('ATOM', 'HETATM')):
                                    try:
                                        res_nums.add(int(line[22:26].strip()))
                                    except ValueError:
                                        continue
                    except Exception as e:
                        print(f"    Warning: Could not parse {atm_file.name}: {e}")
                pocket['residue_numbers'] = sorted(res_nums)
        else:
            print("    Warning: fpocket 'pockets/' output directory not found -- "
                  "residue-level pocket membership unavailable for this run")
            for pocket in pockets:
                pocket['residue_numbers'] = []

        return pockets

    def _geometry_based_detection(self, pdb_content, residue_numbers):
        """Geometry-based pocket detection fallback"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', StringIO(pdb_content))

        pockets = []

        for model in structure:
            for chain in model:
                residues = list(chain.get_residues())

                # Look for hydrophobic surface patches
                for i in range(5, len(residues) - 5):
                    residue = residues[i]

                    if residue.id[0] != ' ':
                        continue

                    # Check if this is a hydrophobic residue on the surface
                    try:
                        if residue.get_resname() not in ['LEU', 'ILE', 'VAL', 'PHE', 'MET']:
                            continue

                        ca = residue['CA']

                        # Calculate local concavity
                        prev_ca = residues[i-3]['CA']
                        next_ca = residues[i+3]['CA']

                        vec1 = ca.coord - prev_ca.coord
                        vec2 = next_ca.coord - ca.coord

                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        angle = np.arccos(np.clip(cos_angle, -1, 1))

                        if angle > 2.0:  # Sharp turn suggests groove
                            center_res = i + 1 if residue_numbers is None else residue_numbers[i]
                            # This fallback has no real cavity geometry, so we
                            # can't recover an exact residue list the way the
                            # fpocket path can. Approximate the pocket as a
                            # +/-4 residue window around the groove center so
                            # downstream residue->pocket lookups (pocket_dict
                            # in app.py, the "CRM1 Pockets" 3D color mode,
                            # combined_score's pocket contribution) have
                            # *something* real to key off of instead of
                            # always being empty.
                            window_lo = max(0, i - 4)
                            window_hi = min(len(residues) - 1, i + 4)
                            if residue_numbers is None:
                                window_res_nums = list(range(window_lo + 1, window_hi + 2))
                            else:
                                window_res_nums = residue_numbers[window_lo:window_hi + 1]
                            pockets.append({
                                'id': len(pockets) + 1,
                                'center_residue': center_res,
                                'residue_numbers': sorted(set(window_res_nums)),
                                'residue_name': residue.get_resname(),
                                'type': 'geometry_based',
                                'method': 'fallback'
                            })

                    except (KeyError, IndexError):
                        continue

        return pockets[:5]

    @staticmethod
    def _pocket_principal_dims(coords):
        """PCA-based (length, width, depth) proxy for a 3D point cloud: the
        extent (max-min projection) along each of the cloud's own principal
        axes, sorted descending. Comparing extents along a shape's OWN
        principal axes (rather than a fixed x/y/z bounding box) means two
        pockets with the same real shape/elongation score as similar
        regardless of how they happen to be oriented in space -- important
        since a candidate protein's fpocket cavity will essentially never
        share CRM1's absolute orientation."""
        coords = np.asarray(coords, dtype=float)
        if len(coords) < 3:
            return None
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        dims = []
        for axis in vt:
            proj = centered @ axis
            dims.append(float(proj.max() - proj.min()))
        return sorted(dims, reverse=True)

    @staticmethod
    def _shape_similarity(dims_a, dims_b):
        """0-1 similarity between two (length, width, depth)-style triples
        (already axis-matched by sorting descending in
        _pocket_principal_dims), via mean relative error per axis."""
        if not dims_a or not dims_b:
            return 0.0
        errs = []
        for a, b in zip(dims_a, dims_b):
            denom = max(a, b, 1e-6)
            errs.append(abs(a - b) / denom)
        return max(0.0, 1.0 - sum(errs) / len(errs))

    @staticmethod
    def _composition_similarity(candidate_resnames):
        """Cosine similarity between a candidate pocket's residue-type
        histogram and CRM1's real 14-residue contact-residue profile
        (CRM1_POCKET_COMPOSITION, from the actual 3NBY <5A-of-NES
        residues) -- restricted to the 6 residue types in that profile so
        polar/charged residues lining the candidate pocket correctly drag
        the score down (they dilute the histogram) instead of being
        ignored, the way a single averaged hydrophobicity number would."""
        if not candidate_resnames:
            return 0.0
        types = list(CRM1_POCKET_COMPOSITION.keys())
        target = np.array([CRM1_POCKET_COMPOSITION[t] for t in types])
        counts = Counter(candidate_resnames)
        total = len(candidate_resnames)
        candidate = np.array([counts.get(t, 0) / total for t in types])
        denom = np.linalg.norm(target) * np.linalg.norm(candidate)
        if denom < 1e-9:
            return 0.0
        return float(np.dot(target, candidate) / denom)

    @staticmethod
    def _axis_ranks(coords):
        """Normalized 0-1 position of each point along a point cloud's own
        longest principal axis (rank 0.0 = one end of the groove, 1.0 = the
        other). Used by _residue_residue_match to give each residue a
        relative position along the groove without needing a real
        structural alignment to CRM1 -- we don't know which end of a
        candidate cavity corresponds to which end of CRM1's groove, so the
        caller tries both directions and keeps whichever scores better."""
        coords = np.asarray(coords, dtype=float)
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[0]
        lo, hi = proj.min(), proj.max()
        if hi - lo < 1e-6:
            return [0.5] * len(coords)
        return list((proj - lo) / (hi - lo))

    @staticmethod
    def _residue_residue_match(candidate_coords, candidate_resnames,
                                template_coords, template_resnames):
        """Position-aware residue-to-residue comparison, added.

        The existing _composition_similarity check is a bag-of-residues
        comparison -- it can't tell a candidate pocket that happens to have
        one bulky Phe and one small Ala (in either order) from CRM1's real
        groove, which specifically has bulky residues (Phe554/561/572) at
        some positions and small ones (Ala538/541) at others (the Güttler et
        al. 2010 sub-pocket size-preference pattern). This pairs each
        candidate residue to the CRM1 contact residue sitting at the
        closest RELATIVE position along the groove's long axis, then
        compares real side-chain volumes (RESIDUE_VOLUME) for that specific
        pair -- so a bulky candidate residue lining up against one of CRM1's
        real bulky-preferring positions scores well, but the same bulky
        residue lining up against one of CRM1's real small-preferring
        positions does not, even though the aggregate composition looks
        identical in both cases.

        Caveat, not real docking: there is no actual structural alignment
        telling us which end of the candidate cavity corresponds to which
        end of CRM1's groove, so both axis directions are tried and the
        better-scoring one is kept. Treat this as a coarse, order-aware
        proxy for the pocket-size pattern, not a literal per-residue docking
        result.

        Returns (score 0-1, list of (candidate_resname, template_resname,
        pair_similarity) for the best-scoring direction).
        """
        if len(candidate_coords) < 2 or len(template_coords) < 2:
            return 0.0, []

        template_ranks = CRM1AwarePocketDetector._axis_ranks(template_coords)
        template_vols = [RESIDUE_VOLUME.get(rn[:3], 130.0) for rn in template_resnames]

        candidate_ranks_fwd = CRM1AwarePocketDetector._axis_ranks(candidate_coords)
        candidate_vols = [RESIDUE_VOLUME.get(rn[:3], 130.0) for rn in candidate_resnames]

        best_score = -1.0
        best_pairs = []
        for candidate_ranks in (candidate_ranks_fwd, [1.0 - r for r in candidate_ranks_fwd]):
            pairs = []
            sims = []
            for c_rank, c_vol, c_name in zip(candidate_ranks, candidate_vols, candidate_resnames):
                j = min(range(len(template_ranks)), key=lambda k: abs(template_ranks[k] - c_rank))
                t_vol = template_vols[j]
                t_name = template_resnames[j]
                # ~100 Å³ spans roughly Ala (88.6) to Phe/Trp (190-228) --
                # the real small-vs-bulky range CRM1's groove actually uses.
                sim = max(0.0, 1.0 - abs(c_vol - t_vol) / 100.0)
                sims.append(sim)
                pairs.append((c_name, t_name, sim))
            avg = sum(sims) / len(sims)
            if avg > best_score:
                best_score = avg
                best_pairs = pairs

        return best_score, best_pairs

    @staticmethod
    def _residue_info_for_numbers(pdb_content, residue_numbers):
        """Real CA coordinates + residue names for a candidate pocket's
        residue_numbers (already recovered per-pocket by
        _parse_fpocket_output's residue-membership fix), parsed from the
        candidate's own structure. pdb_content is already passed into
        _filter_for_crm1_compatibility for every call -- this is what
        actually lets the shape/charge/composition checks below compare
        against real geometry instead of a presence-only flag."""
        if not residue_numbers:
            return [], []
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('candidate', StringIO(pdb_content))
        except Exception:
            return [], []
        wanted = set(residue_numbers)
        coords, resnames = [], []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ' and residue.id[1] in wanted and 'CA' in residue:
                        coords.append(residue['CA'].coord)
                        resnames.append(residue.get_resname())
            break  # first model only -- candidate structures are single-model
        return coords, resnames

    def _filter_for_crm1_compatibility(self, pockets, pdb_content, sequence):
        """
        Apply CRM1-specific filtering based on known binding requirements.

        rewrite: steps 3 and 6 used to be presence-only flags
        that never actually compared a candidate pocket to CRM1's real
        geometry or residue composition -- a flat +0.15 if fpocket merely
        *reported* an elongation/planarity key, and a flat +0.1 if a CRM1
        reference template object had been loaded at all (see this
        function's own old comment, "simplified - could be enhanced with
        structural alignment"). Step 5 (charge density) was previously unimplemented
        with no implementation. All three now use pdb_content (already
        passed into every call) to recover the candidate pocket's real CA
        coordinates and residue identities via _residue_info_for_numbers,
        then compare those against CRM1's actual 14-residue contact
        geometry/composition from 3NBY (_pocket_principal_dims /
        _shape_similarity / _composition_similarity) or a real charge
        count, instead of a boolean flag. Falls back to the old
        flag-based (smaller) bonus when a pocket has no residue_numbers to
        look coordinates up for (e.g. the geometry-based fallback
        detector, or an fpocket run whose pockets/ directory was
        unavailable). reweight (first/second pass): the per-factor weights
        were originally hand-picked, never checked against real labels.
        compute_crm1_factor_stats.py / compute_crm1_joint_weights.py tested
        each factor individually and jointly (standardized logistic
        regression, all 7 together so correlated factors don't
        double-count) against real evaluated examples, and weights were
        reallocated proportional to each POSITIVE-coefficient factor's
        magnitude within a fixed 1.45 total point-budget (so the existing
        >=0.3 pass threshold and _score_to_confidence bands stay
        meaningful); negative-coefficient factors are zeroed rather than
        flipped to a penalty, since "doesn't help" is much better supported
        by these sample sizes than "actively harmful" is -- still computed
        into crm1_subscores for monitoring either way. (third pass): re-fit again after the leucine_zipper
        hard-negative sample was refreshed via a RunPod run of
        evaluate_crm1_pocket_signal.py (445 examples with all 7 factors,
        111 positive / 334 negative -- see compute_crm1_joint_weights.py's
        printed output / crm1_joint_weights.json). Coefficients: volume_A3
        +0.412, shape_similarity +0.396, charge_score +0.257, druggability
        +0.038 (all kept, weighted by magnitude); hydrophobicity -0.019,
        composition_similarity -0.247, residue_residue_score -0.303 (all
        zeroed). This is the first pass where hydrophobicity's own
        coefficient came out genuinely negative rather than a token
        positive -- it was already contributing 0 to score either way, but
        now that's for the same reason the other two zeroed factors are,
        not a separate "doesn't matter" rationale. Re-run
        compute_crm1_joint_weights.py and update the weights below again
        any time the eval sample composition changes meaningfully -- do not
        treat this pass's numbers as final either. See
        CRM1_pocket_scoring_evaluation_2026-07-27.md for the original
        writeup (predates this third pass).
        """
        if not pockets:
            return []

        scored_pockets = []
        start_time = time.time()
        timed_out = False

        for i, pocket in enumerate(pockets):
            # Wall-clock guard, see __init__'s pocket_filter_timeout
            # comment for why this lives here rather than as a signal.alarm
            # around the whole call. Checked once per pocket rather than
            # once per call so a structure with many candidate pockets
            # (e.g. Myosin-9, 1960 residues) can't run unbounded even though
            # each individual pocket's scoring is fast.
            if time.time() - start_time > self.pocket_filter_timeout:
                print(f"Warning: CRM1-compatibility scoring exceeded {self.pocket_filter_timeout}s "
                      f"({i}/{len(pockets)} pockets scored) -- returning partial results "
                      f"instead of hanging. Likely an unusually large/elongated structure "
                      f"with many candidate pockets.")
                timed_out = True
                break

            score = 0.0
            reasons = []
            # Raw, unweighted per-factor values (0-1 where
            # applicable, None where that factor wasn't computable for this
            # pocket), independent of the hand-picked weights below --
            # AUC/correlation against real labels is scale-invariant to a
            # positive weight, so testing these raw values directly (via
            # evaluate_crm1_pocket_signal.py) tells you whether each factor
            # of crm1_compatibility_score actually carries signal on its
            # own, same way the fpocket-vs-burial blend question got
            # answered empirically instead of guessed.
            subscores = {
                'volume_A3': None, 'hydrophobicity': None, 'shape_similarity': None,
                'druggability': None, 'charge_score': None, 'composition_similarity': None,
                'residue_residue_score': None,
            }

            # 1. Volume check (CRM1 groove: 800-1500 Å³)
            volume = pocket.get('volume', pocket.get('total_sasa', 0))
            subscores['volume_A3'] = volume
            if 200 <= volume <= 2500:
                # (third pass): re-fit against the RunPod-refreshed
                # crm1_eval_results.json (445 examples w/ all 7 factors,
                # 111 positive / 334 negative -- denser leucine_zipper
                # coverage than either earlier pass). Joint coefficient
                # +0.412, now the single strongest factor (narrowly ahead of
                # shape_similarity's +0.396). See compute_crm1_joint_weights.py
                # / crm1_joint_weights.json. Optimal/baseline ratio (4/3)
                # held constant across all three reweighting passes.
                volume_score = 0.542
                if 800 <= volume <= 1500:
                    volume_score = 0.723  # Optimal
                score += volume_score
                reasons.append(f"Volume: {volume:.0f} Å³")

            # Real per-residue coordinates/identities for this candidate
            # pocket, when fpocket's residue-level output was available.
            # Moved above step 2 (was below it) so the hydrophobicity check
            # can use these residue identities directly -- see the real bug
            # fixed below.
            residue_numbers = pocket.get('residue_numbers') or []
            coords, resnames = self._residue_info_for_numbers(pdb_content, residue_numbers)

            # 2. Hydrophobicity (critical for NES binding)
            #
            # Real bug found via evaluate_crm1_pocket_signal.py:
            # this used to read pocket['hydrophobicity_score'] directly from
            # real fpocket's own output (structure_info.txt's "Hydrophobicity
            # score" line, parsed as-is in _parse_fpocket_output) and compare
            # it to 0.35 as if it were a 0-1 ratio. It isn't -- fpocket's own
            # hydrophobicity score is an unnormalized internal metric that
            # routinely comes out in the 10s-50s range, not 0-1. That meant
            # this >=0.35 check was almost always true regardless of whether
            # the pocket was actually hydrophobic (silently non-discriminative
            # for real fpocket runs), AND -- more consequentially -- app.py's
            # crm1_binding_affinity = pocket['score'] * pocket['hydrophobic_ratio']
            # was multiplying by this same unnormalized 10-50 value instead of
            # a 0-1 fraction, so crm1_binding_affinity has been wildly outside
            # its assumed ~0-1 range (confirmed empirically: real evaluation
            # run showed fpocket-based affinity averaging ~18-32, not ~0-1)
            # every time real fpocket ran. The geometry-based fallback never
            # set this key at all (always defaulted to 0), so this only
            # showed up with real fpocket installed -- which is why it went
            # unnoticed in environments without fpocket installed.
            #
            # Fixed by computing a REAL, bounded-by-construction 0-1 fraction
            # directly from this pocket's own lining-residue identities
            # (already recovered above via _residue_info_for_numbers), same
            # approach already used for the charge-density and
            # residue-composition checks below. Overwrites
            # pocket['hydrophobicity_score'] in place, so app.py's
            # 'hydrophobic_ratio' (read from this same dict later) picks up
            # the corrected value automatically -- no separate app.py fix
            # needed.
            if resnames:
                hydro = sum(1 for r in resnames if r in HYDROPHOBIC_RESNAMES) / len(resnames)
                pocket['hydrophobicity_score'] = hydro
            else:
                # No residue-level data available (e.g. geometry fallback, or
                # an fpocket run whose pockets/ directory was missing) --
                # nothing real to compute from, so fall back to whatever was
                # already there (0 if absent), same as before this fix.
                hydro = pocket.get('hydrophobicity_score', 0)
            subscores['hydrophobicity'] = hydro
            # (third pass): joint coefficient is now -0.019 on
            # the RunPod-refreshed sample (n=445) -- actually negative for
            # the first time (was a functionally-zero +0.001 on the prior
            # pass), so this stays zeroed under the same rule negative-
            # coefficient factors get elsewhere (composition_similarity,
            # residue_residue_score below). Still computed/reported for
            # monitoring.
            if hydro >= 0.35:
                score += 0.0
                reasons.append(f"Hydrophobic: {hydro:.2f} (not scored -- see comment)")

            # 3. Shape -- real PCA-based comparison to CRM1's actual groove
            # dimensions when we have real coordinates; falls back to
            # fpocket's own elongation/planarity flag (weaker -- no
            # coordinate comparison happens in the fallback) otherwise.
            template_dims = self.crm1_pocket_template.get('tight_dims') if self.crm1_pocket_template else None
            if len(coords) >= 3 and template_dims:
                candidate_dims = self._pocket_principal_dims(coords)
                shape_sim = self._shape_similarity(candidate_dims, template_dims)
                subscores['shape_similarity'] = shape_sim
                # (third pass): joint coefficient +0.396 on the
                # RunPod-refreshed sample (n=445) -- narrowly edged out by
                # volume_A3's +0.412 this pass (was the single strongest
                # factor on the prior, smaller sample). Still one of the
                # two dominant factors.
                score += 0.521 * shape_sim
                reasons.append(f"Shape match to CRM1 groove: {shape_sim:.2f}")
            elif 'pocket_elongation' in pocket or 'planarity' in pocket:
                score += 0.278  # scaled with shape_similarity's reweight above (~53% ratio,
                                 # held constant across all three reweighting passes)
                reasons.append("Elongated shape (fpocket flag only, no coordinate comparison)")

            # 4. Surface accessibility
            drug_score = pocket.get('druggability_score', pocket.get('drug_score', 0))
            subscores['druggability'] = drug_score
            if drug_score >= 0.25:
                score += 0.050  # (third pass): joint coefficient +0.038 on the
                                 # RunPod-refreshed sample (n=445) -- down from +0.101, still
                                 # weakly positive so it keeps a small, now smaller, share
                                 # rather than being zeroed like the negative-coefficient factors
                reasons.append(f"Accessible: {drug_score:.2f}")

            # 5. Low charge density (NES binding groove has low charge) --
            # previously unimplemented. None of CRM1's real 14
            # contact residues are charged, so we score how close the
            # candidate pocket gets to that (full credit at 0% charged
            # residues, none at >=30%).
            if resnames:
                charged_frac = sum(1 for r in resnames if r in CHARGED_RESNAMES) / len(resnames)
                charge_score = max(0.0, 1.0 - charged_frac / 0.3)
                subscores['charge_score'] = charge_score
                # (third pass): joint coefficient +0.257 on the
                # RunPod-refreshed sample (n=445, denser leucine_zipper
                # coverage still) -- consistent with the second pass's
                # finding that this factor separates real NES from
                # coiled_coil but only weakly/inconsistently from the
                # harder leucine_zipper case (see plot_crm1_factors_boxplot.py's
                # output: AUC vs. leucine_zipper is not significant on this
                # sample, AUC vs. coiled_coil still is). Kept as the
                # third-largest weight rather than zeroed since the joint
                # coefficient is still clearly positive even though the
                # marginal picture is mixed.
                score += 0.338 * charge_score
                reasons.append(f"Charge density: {charged_frac:.0%} charged residues")

            # 6. Residue-composition match to CRM1's real 14-residue
            # contact profile (Phe/Leu/Met/Ile/Val/Cys), replacing the old
            # flat "template loaded, so +0.1" bonus. Uses
            # CRM1_POCKET_COMPOSITION directly, so this works even when no
            # reference PDB was loaded at all (self.crm1_pocket_template
            # is None) -- it only needs the candidate's own residue
            # identities.
            #
            # Weight zeroed (was 0.15/0.05 fallback). Joint
            # standardized logistic regression gave this factor a NEGATIVE
            # coefficient both on the original fit (n=269, coef -0.19) and
            # the re-fit after growing the leucine_zipper sample (n=331,
            # coef -0.304, if anything more negative) -- consistently
            # backwards, hard negatives scoring marginally higher than
            # real NES on this specific measure. Still computed into
            # crm1_subscores for monitoring; not zeroed to a penalty since
            # "doesn't help" is better supported than "actively harmful"
            # at this sample size. See compute_crm1_factor_stats.py.
            if resnames:
                comp_sim = self._composition_similarity(resnames)
                subscores['composition_similarity'] = comp_sim
                score += 0.0 * comp_sim
                reasons.append(f"Residue-type match to CRM1 groove: {comp_sim:.2f} (not scored -- see comment)")
            elif self.crm1_pocket_template:
                score += 0.0
                reasons.append("Template loaded (no residue-level comparison possible)")

            # 7. Residue-residue positional match (added --
            # pairs each candidate residue to whichever CRM1 contact residue
            # sits at the closest relative position along the groove axis
            # and compares real side-chain volumes for that specific pair,
            # instead of only the order-agnostic bag-of-residues check
            # above. See _residue_residue_match's docstring for the
            # axis-direction caveat (not real docking).
            #
            # Weight zeroed one session after being added.
            # Original fit (n=269): marginal AUC=0.419 (backwards, not
            # significant, p=0.119), joint coefficient -0.24. Re-fit after
            # growing the leucine_zipper sample (n=331): marginal
            # AUC=0.405, joint coefficient -0.253 -- consistently
            # backwards on both fits. Kept computed/reported for continued
            # monitoring -- the axis-direction heuristic this relies on
            # (no real docking) is a plausible reason it might not be
            # working yet, not necessarily that the underlying idea is
            # wrong. See compute_crm1_factor_stats.py.
            template_coords = self.crm1_pocket_template.get('tight_coords') if self.crm1_pocket_template else None
            template_resnames = self.crm1_pocket_template.get('matched_residues') if self.crm1_pocket_template else None
            if coords and len(coords) >= 2 and template_coords is not None and template_resnames:
                rr_score, rr_pairs = self._residue_residue_match(
                    coords, resnames, template_coords, template_resnames
                )
                subscores['residue_residue_score'] = rr_score
                score += 0.0 * rr_score
                reasons.append(f"Residue-residue positional match to CRM1 groove: {rr_score:.2f} (not scored -- see comment)")
                if rr_pairs:
                    ranked_pairs = sorted(rr_pairs, key=lambda p: -p[2])
                    for c_name, t_name, sim in ranked_pairs[:3]:
                        reasons.append(f"    best size match: {c_name} <-> {t_name} ({sim:.2f})")
                    for c_name, t_name, sim in ranked_pairs[-3:]:
                        if len(ranked_pairs) > 3:
                            reasons.append(f"    weak size match: {c_name} <-> {t_name} ({sim:.2f})")

            # Only keep compatible pockets
            if score >= 0.3:
                pocket['crm1_compatibility_score'] = round(score, 2)
                pocket['crm1_compatibility_reasons'] = reasons
                pocket['crm1_subscores'] = subscores
                pocket['confidence'] = self._score_to_confidence(score)
                scored_pockets.append(pocket)

        scored_pockets.sort(key=lambda x: x.get('crm1_compatibility_score', 0), reverse=True)

        return scored_pockets

    def _score_to_confidence(self, score):
        """Convert numerical score to confidence level"""
        if score >= 0.7:
            return 'very_high'
        elif score >= 0.55:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'


# Convenience function
def detect_crm1_pockets(pdb_content, crm1_reference_path=None, sequence=None):
    """
    Quick function to detect CRM1-compatible pockets

    Args:
        pdb_content: PDB file content as string
        crm1_reference_path: Optional path to CRM1 reference structure (3NBY)
        sequence: Optional protein sequence for context

    Returns:
        List of pocket dictionaries with CRM1 compatibility scores
    """
    detector = CRM1AwarePocketDetector(crm1_reference_path)
    return detector.detect_pockets(pdb_content, sequence=sequence)
