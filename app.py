from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import re
import time
import os
import json
from Bio.PDB import PDBParser
from io import StringIO
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from sumoylation_predictor import SUMOylationPredictor
from quick_helix_analysis import quick_structural_analysis, batch_quick_analysis

# Handle different BioPython versions
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    def three_to_one(residue_name):
        """Convert three-letter amino acid code to one-letter code"""
        three_to_one_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return three_to_one_dict.get(residue_name.upper(), 'X')

# localCIDER (linear hydropathy / NCPR / FCR profiles) -- optional import so
# the app still starts if it isn't installed yet; every caller checks
# 'cider_computed' rather than assuming these arrays are ever real.
try:
    from localcider.sequenceParameters import SequenceParameters
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False
    print("WARNING: localcider not installed -- linear hydropathy/NCPR/FCR "
          "features disabled. Install with: pip install localcider")

_STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def compute_linear_cider_profiles(sequence, blob_len=5):
    """
    Real localCIDER sliding-window profiles: linear hydropathy, linear NCPR
    (net charge per residue), linear FCR (fraction of charged residues), and
    linear (Wootton-Federhen) sequence complexity -- the 4 profiles CIDER's
    own web tool plots by default. localCIDER handles edge-window padding
    internally, so each returned array is exactly len(sequence) long -- one
    value per residue, directly usable for per-residue structure coloring or
    line-chart plotting.

    Returns dict with 'positions' (1-based), 'linear_hydropathy',
    'linear_ncpr', 'linear_fcr', 'linear_complexity', and 'cider_computed'
    (False if localcider isn't installed or the calculation fails for this
    sequence -- in which case the arrays are flat placeholders and MUST NOT
    be treated as real).
    """
    n = len(sequence)
    if not CIDER_AVAILABLE or n == 0:
        return {
            'positions': list(range(1, n + 1)),
            'linear_hydropathy': [0.0] * n,
            'linear_ncpr': [0.0] * n,
            'linear_fcr': [0.0] * n,
            'linear_complexity': [0.0] * n,
            'cider_computed': False,
        }
    try:
        # localCIDER only accepts the 20 standard amino acids -- substitute
        # anything else (rare 'X' from unresolved residues etc.) with the
        # most neutral placeholder (Gly) rather than crashing the whole
        # calculation over a handful of residues.
        clean_seq = ''.join(aa if aa in _STANDARD_AA else 'G' for aa in sequence.upper())
        sp = SequenceParameters(clean_seq)
        eff_blob = min(blob_len, n if n % 2 == 1 else n - 1)
        eff_blob = max(1, eff_blob)
        _, hydropathy = sp.get_linear_hydropathy(blobLen=eff_blob)
        _, ncpr = sp.get_linear_NCPR(blobLen=eff_blob)
        _, fcr = sp.get_linear_FCR(blobLen=eff_blob)
        try:
            # NOTE: unlike hydropathy/NCPR/FCR (which localCIDER internally
            # edge-pads to exactly length n), get_linear_complexity returns
            # only the positions where a full blobLen-wide window fits (n -
            # blobLen + 1 values), with its own real 1-based x positions --
            # e.g. for n=35/blobLen=5 it covers positions 3..33, not 1..35.
            # Confirmed by direct testing. Left as-is this silently
            # misaligns against 'positions'/the other 3 arrays for anything
            # downstream that zips them by index (like a line chart), so we
            # scatter the real values back onto their real positions and
            # edge-pad (constant/replicate) the remainder to match length n.
            x_complexity, complexity_vals = sp.get_linear_complexity(blobLen=eff_blob)
            complexity = [0.0] * n
            if len(complexity_vals):
                for xi, ci in zip(x_complexity, complexity_vals):
                    idx = int(round(float(xi))) - 1
                    if 0 <= idx < n:
                        complexity[idx] = float(ci)
                first_idx = max(0, int(round(float(x_complexity[0]))) - 1)
                last_idx = min(n - 1, int(round(float(x_complexity[-1]))) - 1)
                first_val, last_val = float(complexity_vals[0]), float(complexity_vals[-1])
                for i in range(0, first_idx):
                    complexity[i] = first_val
                for i in range(last_idx + 1, n):
                    complexity[i] = last_val
        except Exception:
            complexity = [0.0] * n
        return {
            'positions': list(range(1, n + 1)),
            'linear_hydropathy': [float(v) for v in hydropathy],
            'linear_ncpr': [float(v) for v in ncpr],
            'linear_fcr': [float(v) for v in fcr],
            'linear_complexity': complexity,
            'cider_computed': True,
        }
    except Exception as e:
        print(f"WARNING: localCIDER linear profile calculation FAILED "
              f"(sequence length {n}): {e}")
        return {
            'positions': list(range(1, n + 1)),
            'linear_hydropathy': [0.0] * n,
            'linear_ncpr': [0.0] * n,
            'linear_fcr': [0.0] * n,
            'linear_complexity': [0.0] * n,
            'cider_computed': False,
        }

# ============================================================================
# Flask app initialization
# ============================================================================
app = Flask(__name__)

# Configure CORS with explicit settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5000", "http://127.0.0.1:3000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Add after_request handler for additional CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ============================================================================
# ML Enhancement Initialization - WITH CRM1 REFERENCE
# ============================================================================

print("\n" + "="*70)
print("Initializing AlphaFold NES Detector with ML Enhancements")
print("="*70)

# Define CRM1 reference path - HANDLES BOTH WINDOWS AND WSL
import platform

# Detect if we're in WSL
def is_wsl():
    return 'microsoft' in platform.uname().release.lower() or 'wsl' in platform.uname().release.lower()

# Set up paths based on environment
#
# IMPORTANT: CRM1_Ran_only.pdb (chains A+C only: CRM1 + RanGTP, produced by
# extract_crm1_ran.py) is the correct reference structure - it has the NES
# groove exposed. The plain crm1.pdb fallback further down is the raw,
# uncleaned crystal structure (6 chains, including a bound cargo protein
# still occupying the NES groove) and should only be used if the cleaned
# file is ever missing.
# Resolved relative to this file and to the working directory, so the same
# code runs unchanged on Windows, WSL, macOS and a bare Linux compute node.
REPO_ROOT = Path(__file__).resolve().parent

CRM1_PATH_OPTIONS = [
    str(REPO_ROOT / 'crm1_reference' / 'CRM1_Ran_only.pdb'),
    str(Path.cwd() / 'crm1_reference' / 'CRM1_Ran_only.pdb'),
    './crm1_reference/CRM1_Ran_only.pdb',
    str(REPO_ROOT / 'crm1.pdb'),
    str(Path.cwd() / 'crm1.pdb'),
    './crm1.pdb',
    str(REPO_ROOT / 'CRM1.pdb'),
    str(Path.cwd() / 'CRM1.pdb'),
]
print(f"Platform: {'WSL' if is_wsl() else platform.system()}")

# Check if CRM1 reference exists
print(f"\nChecking for CRM1 structure...")
print(f"   Current directory: {Path.cwd()}")
crm1_ref_found = None

for path_option in CRM1_PATH_OPTIONS:
    print(f"   Trying: {path_option}")
    try:
        path_obj = Path(path_option)
        if path_obj.exists() and path_obj.is_file():
            crm1_ref_found = str(path_obj.absolute())
            print(f"   FOUND!")
            break
        else:
            print(f"   Not found")
    except Exception as e:
        print(f"   Error: {e}")

if crm1_ref_found:
    print(f"\nCRM1 structure loaded: {crm1_ref_found}")
    print(f"   File size: {Path(crm1_ref_found).stat().st_size / 1024:.1f} KB")
    CRM1_REFERENCE_PATH = crm1_ref_found
else:
    print(f"\nFailed: CRM1 structure NOT found in any expected location!")
    print(f"   Searched:")
    for p in CRM1_PATH_OPTIONS:
        print(f"      - {p}")
    print(f"\n   Expected file: {REPO_ROOT / 'crm1_reference' / 'CRM1_Ran_only.pdb'}")
    print(f"   Run setup_crm1_reference.py to build the cleaned reference structure,")
    print(f"   or place crm1.pdb in the repository root.")
    CRM1_REFERENCE_PATH = None
print()

# Initialize ML components
try:
    # CORRECTED IMPORTS - Use the actual file names
    # Option 1: If you renamed the file
    from pocket_detector import CRM1AwarePocketDetector as PocketDetector

    # Option 2: If you kept the original filename, use this instead:
    # from pocket_detector_ENHANCED import CRM1AwarePocketDetector as PocketDetector

    # UPDATED: Use improved NES predictor with LocNES + NESmapper features
    from nes_ml_predictor_improved import ImprovedNESPredictor

    # Initialize ML predictor (always works)
    ml_predictor = ImprovedNESPredictor()
    print("Improved ML NES predictor initialized (LocNES + NESmapper)")
    print("  PSSM rank-based scoring enabled")
    print("  Flanking region analysis enabled")
    print("  Class 3 NES detection enabled")

    print("\nInitializing SUMOylation Predictor...")
    sumo_predictor = SUMOylationPredictor()
    print("SUMOylation predictor ready")

    print("\nInitializing NLS Predictor...")
    # Own try/except: an NLS-side problem (missing model, bad deps) must
    # never take down the already-working NES/SUMOylation predictors above,
    # which a shared try/except around this whole block would otherwise do.
    try:
        from nls_ml_predictor import NLSPredictor
        nls_predictor = NLSPredictor()
        if nls_predictor.model is not None:
            print(f"NLS predictor ready (model: {nls_predictor.model_name})")
        else:
            print("Warning: NLS predictor loaded but no trained model found in models_nls/ "
                  "-- run `python nls_ml_predictor.py train` first")
    except Exception as nls_e:
        print(f"Warning: NLS predictor initialization failed: {nls_e}")
        nls_predictor = None

    # Initialize pocket detector with CRM1 reference if available
    if crm1_ref_found:
        print(f"CRM1 reference found: {crm1_ref_found}")
        pocket_detector = PocketDetector(crm1_reference_path=crm1_ref_found)
        print("Pocket detector initialized WITH CRM1 template matching")
        print("  Enhanced CRM1-specific pocket detection enabled")
    else:
        print("Warning: CRM1 reference not found - using geometry-based fallback")
        print("  To enable CRM1 template matching:")
        print("     1. Run: python setup_crm1_reference.py")
        print("     2. Restart this server")
        pocket_detector = PocketDetector()
        print("Pocket detector initialized in fallback mode")

    print("\n" + "="*70)
    print("ENHANCEMENT STATUS")
    print("="*70)
    print("Machine Learning NES Prediction:  ENABLED")
    print(f"{'' if crm1_ref_found else 'Warning: '} CRM1 Template Pocket Matching:     {'ENABLED' if crm1_ref_found else 'DISABLED (no reference)'}")
    print("fpocket Integration:               ENABLED")
    print("="*70 + "\n")

except ImportError as e:
    print(f"Warning: Import error: {e}")
    print("\nTroubleshooting:")
    print("  1. Check that pocket_detector.py exists (or pocket_detector_ENHANCED.py)")
    print("  2. Check that nes_ml_predictor.py exists")
    print("  3. Install required packages: pip install scikit-learn joblib")
    print("\nFalling back to basic mode (no ML enhancements)")
    ml_predictor = None
    pocket_detector = None
    nls_predictor = None
    print("="*70 + "\n")

except Exception as e:
    print(f"Warning: Initialization error: {e}")
    import traceback
    traceback.print_exc()
    print("\nFalling back to basic mode")
    ml_predictor = None
    pocket_detector = None
    nls_predictor = None
    print("="*70 + "\n")

# ============================================================================
# MD JOB QUEUE INITIALIZATION
# ============================================================================

try:
    from md_job_queue import get_job_queue
    import remote_md_dispatch
    job_queue = get_job_queue(crm1_pdb_path=crm1_ref_found)  # Pass CRM1 path
    print("MD Job Queue initialized")
    if remote_md_dispatch.REMOTE_ENABLED:
        print(f"  MD simulations will run REMOTELY on "
              f"{remote_md_dispatch.REMOTE_USER}@{remote_md_dispatch.REMOTE_HOST}")
        if not remote_md_dispatch.check_remote_ready():
            print("  Warning: Could not verify remote host is ready (SSH key auth working + "
                  "remote_md_runner.py present in MD_REMOTE_WORKDIR). Jobs will fall "
                  "back to local execution per-candidate if remote dispatch fails.")
    elif crm1_ref_found:
        print(f"  MD simulations will use CRM1 structure locally: {crm1_ref_found}")
    else:
        print("  Warning: No CRM1 structure - MD will use sequence-based estimates")
    MD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MD Job Queue not available: {e}")
    print("  Install OpenMM: conda install -c conda-forge openmm pdbfixer")
    job_queue = None
    MD_AVAILABLE = False
except Exception as e:
    print(f"Warning: MD initialization error: {e}")
    job_queue = None
    MD_AVAILABLE = False

# ============================================================================
# Continue with the rest of your app.py from line 55 onwards
# (Hydrophobicity scale, routes, etc.)
# ============================================================================


# Hydrophobicity scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0
}

# Charge at pH 7
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0
}

def get_alphafold_structure_info(uniprot_id):
    """
    Get AlphaFold structure information using the official API v4
    Returns: list of entries (some proteins have multiple fragments)
    """
    # AlphaFold API v4 endpoint
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

    print(f"Querying AlphaFold API: {api_url}")

    try:
        response = requests.get(api_url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # API returns a list of entries (some proteins have multiple fragments)
            if isinstance(data, list) and len(data) > 0:
                print(f"Found {len(data)} AlphaFold entry/entries for {uniprot_id}")
                return data  # Return all entries
            elif isinstance(data, dict):
                print(f"Found AlphaFold entry for {uniprot_id}")
                return [data]  # Wrap single entry in list
            else:
                print(f"Failed: Unexpected API response format")
                return None
        else:
            print(f"Failed: API returned status {response.status_code}")
            return None

    except Exception as e:
        print(f"Failed: API error: {e}")
        return None

def get_pdb_structures_for_uniprot(uniprot_id):
    """
    Find experimentally solved structures (X-ray/cryo-EM/NMR) for a UniProt
    accession by reading the PDB cross-references embedded in the UniProt
    entry itself (no extra API key / query language needed).
    Returns a list of dicts: {pdb_id, method, resolution, chains}
    """
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    try:
        response = requests.get(uniprot_url, timeout=10)
        if response.status_code != 200:
            print(f"Failed: UniProt lookup failed for {uniprot_id} (status {response.status_code})")
            return []

        data = response.json()
        pdb_entries = []

        for xref in data.get('uniProtKBCrossReferences', []):
            if xref.get('database') != 'PDB':
                continue

            pdb_id = xref.get('id')
            method = None
            resolution = None
            chains = None

            for prop in xref.get('properties', []):
                key = prop.get('key')
                value = prop.get('value')
                if key == 'Method':
                    method = value
                elif key == 'Resolution':
                    resolution = value
                elif key == 'Chains':
                    chains = value

            if pdb_id:
                pdb_entries.append({
                    'pdb_id': pdb_id,
                    'method': method or 'Unknown',
                    'resolution': resolution,
                    'chains': chains,
                })

        print(f"Found {len(pdb_entries)} experimental PDB entry/entries for {uniprot_id}")
        return pdb_entries

    except Exception as e:
        print(f"Error fetching PDB cross-references for {uniprot_id}: {e}")
        return []

@app.route('/api/search', methods=['GET'])
def search_proteins():
    """Search for proteins in AlphaFold database"""
    query = request.args.get('query', '')
    organism = request.args.get('organism', '')

    try:
        # Search UniProt for proteins
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query={query}"
        if organism.lower() != 'all':
            organism_map = {
                'human': '9606',
                'rat': '10116',
                'mouse': '10090',
                'yeast': '559292',
                'e.coli': '83333'
            }
            tax_id = organism_map.get(organism.lower(), '')
            if tax_id:
                uniprot_url += f"+AND+organism_id:{tax_id}"

        uniprot_url += "&format=json&size=50"

        response = requests.get(uniprot_url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = []

            for entry in data.get('results', [])[:20]:  # Limit to 20 results
                accession = entry.get('primaryAccession', '')
                protein_name = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')
                organism_name = entry.get('organism', {}).get('scientificName', 'Unknown')
                sequence_length = entry.get('sequence', {}).get('length', 0)

                results.append({
                    'id': accession,
                    'name': protein_name,
                    'organism': organism_name,
                    'residues': sequence_length,
                    'alphafold_id': f"AF-{accession}-F1"
                })

            return jsonify(results)
        else:
            return jsonify([])

    except Exception as e:
        print(f"Error searching: {e}")
        return jsonify([])

@app.route('/api/models/<protein_id>', methods=['GET'])
def get_models(protein_id):
    """Get AlphaFold models for a protein using API v4"""
    try:
        print(f"\n{'='*60}")
        print(f"Fetching AlphaFold structures for: {protein_id}")
        print(f"{'='*60}")

        # Which structure source(s) to fetch: 'alphafold' (default), 'experimental', or 'both'
        source = request.args.get('source', 'alphafold').lower()
        include_alphafold = source in ('alphafold', 'both')
        include_experimental = source in ('experimental', 'both')

        # Query AlphaFold API (returns list of entries)
        entries = get_alphafold_structure_info(protein_id) if include_alphafold else None

        if include_alphafold and not entries:
            print(f"Warning: No AlphaFold structure found for {protein_id}")
            print(f"This protein may not be in the AlphaFold database yet.")
            print(f"Check: https://alphafold.ebi.ac.uk/entry/{protein_id}")
            entries = []

        # Extract information from API response
        models = []

        # Process each entry (some proteins have multiple fragments/isoforms)
        for entry_idx, entry in enumerate(entries or []):
            entry_id_base = entry.get('entryId', f"AF-{protein_id}-F{entry_idx+1}")
            fragment = entry.get('uniprotStart', 'N/A')
            fragment_end = entry.get('uniprotEnd', 'N/A')
            gene = entry.get('gene', 'Unknown')

            print(f"\nEntry {entry_idx + 1}: {entry_id_base}")
            print(f"  Fragment: residues {fragment}-{fragment_end}")
            print(f"  Gene: {gene}")

            # Debug: print the entire entry to see what fields are available
            print(f"  DEBUG - Entry keys: {list(entry.keys())}")
            print(f"  DEBUG - latestVersion: {entry.get('latestVersion', 'NOT FOUND')}")
            print(f"  DEBUG - allVersions: {entry.get('allVersions', 'NOT FOUND')}")

            # Get all available versions from the API
            # If allVersions is provided, use it; otherwise probe for versions 1-4
            all_versions = entry.get('allVersions')

            if not all_versions:
                # If API doesn't provide version list, try versions 1-4
                # AlphaFold typically has up to 4 versions
                latest_version = entry.get('latestVersion', 4)
                all_versions = list(range(1, latest_version + 1))
                print(f"  allVersions not in API response, probing versions 1-{latest_version}")

            print(f"  Available versions to try: {all_versions}")

            # Fetch each version for this entry
            for version in all_versions:
                print(f"\n  Processing version {version}...")

                # Construct URLs for this version
                pdb_url = f"https://alphafold.ebi.ac.uk/files/{entry_id_base}-model_v{version}.pdb"
                cif_url = f"https://alphafold.ebi.ac.uk/files/{entry_id_base}-model_v{version}.cif"

                # For the latest version, use the URLs from API if available
                if version == entry.get('latestVersion'):
                    pdb_url = entry.get('pdbUrl', pdb_url)
                    cif_url = entry.get('cifUrl', cif_url)

                print(f"    PDB URL: {pdb_url}")

                # Try to download and analyze the structure
                try:
                    print(f"    Downloading...")
                    response = requests.get(pdb_url, timeout=30)

                    if response.status_code == 200:
                        structure_content = response.text

                        # Verify it's PDB format
                        if 'ATOM' not in structure_content and 'HETATM' not in structure_content:
                            print(f"    Failed: Not valid PDB format")
                            continue

                        # Calculate average pLDDT and EXACT number of residues
                        avg_confidence, num_residues = calculate_pdb_stats(structure_content)

                        model_id = f"{entry_id_base}-model_v{version}"

                        # Create descriptive name
                        if len(entries) > 1:
                            model_name = f"Fragment {fragment}-{fragment_end} (v{version})"
                        else:
                            model_name = f"Version {version}"

                        models.append({
                            'model_id': model_id,  # Use full model_id
                            'id': model_id,
                            'version': version,
                            'latestVersion': version,  # Add this
                            'name': model_name,
                            'fragment': f"{fragment}-{fragment_end}",
                            'numResidues': num_residues,  # Actual residue count
                            'length': num_residues,  # Alternative name
                            'avgConfidence': round(avg_confidence, 2),
                            'confidenceAvgLocalScore': round(avg_confidence, 2),  # Alternative name
                            'avg_confidence': round(avg_confidence, 2),
                            'url': pdb_url,
                            'pdbUrl': pdb_url,
                            'cifUrl': cif_url,
                            'gene': gene,
                            'entryId': entry_id_base,  # Add base entry ID
                            'source': 'alphafold'
                        })

                        print(f"    Added version {version}")
                        print(f"      Avg confidence: {avg_confidence:.2f}")
                    else:
                        print(f"    Failed to download (status {response.status_code})")

                except Exception as download_err:
                    print(f"    Failed: Download error: {download_err}")
                    continue

        # Sort AlphaFold models by confidence (highest first), then by version (newest first)
        models.sort(key=lambda x: (-x['avg_confidence'], -x['version']))

        # Fetch experimentally solved (X-ray/cryo-EM/NMR) structures, if requested
        experimental_models = []
        if include_experimental:
            print(f"\nFetching experimental PDB structures for: {protein_id}")
            pdb_entries = get_pdb_structures_for_uniprot(protein_id)

            for pdb_entry in pdb_entries:
                pdb_id = pdb_entry['pdb_id']
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                experimental_models.append({
                    'model_id': f"PDB-{pdb_id}",
                    'id': f"PDB-{pdb_id}",
                    'source': 'experimental',
                    'name': f"{pdb_id} ({pdb_entry['method']})",
                    'method': pdb_entry['method'],
                    'resolution': pdb_entry['resolution'],
                    'chains': pdb_entry['chains'],
                    'entryId': pdb_id,
                    'pdbUrl': pdb_url,
                    'url': pdb_url,
                })

            # Sort by resolution (lower = better); entries without a numeric
            # resolution (e.g. NMR structures) sort to the end
            def _resolution_key(m):
                try:
                    return float(m['resolution'])
                except (TypeError, ValueError):
                    return float('inf')
            experimental_models.sort(key=_resolution_key)

            print(f"Added {len(experimental_models)} experimental structure(s)")

        # Combine: AlphaFold predictions first, then experimental structures
        models = models + experimental_models

        print(f"\n{'='*60}")
        print(f"Total models loaded: {len(models)} "
              f"(AlphaFold: {len(models) - len(experimental_models)}, Experimental: {len(experimental_models)})")
        print(f"{'='*60}\n")

        if len(models) == 0:
            print("Warning: No models could be loaded")
            print("  This might mean the older versions are not available")

        return jsonify(models)

    except Exception as e:
        print(f"Error fetching models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])

def calculate_avg_plddt(structure_content):
    """Calculate average pLDDT from B-factor column in PDB"""
    plddt_values = []

    for line in structure_content.split('\n'):
        if line.startswith('ATOM'):
            # B-factor is in columns 61-66
            try:
                b_factor = float(line[60:66].strip())
                plddt_values.append(b_factor)
            except:
                pass

    if plddt_values:
        return np.mean(plddt_values)
    return 0.0

def calculate_pdb_stats(structure_content):
    """
    Calculate both average pLDDT and EXACT residue count from PDB
    Fixes residue count mismatch between model list and structure view
    """
    from Bio.PDB import PDBParser
    from io import StringIO

    plddt_values = []
    residue_count = 0

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('temp', StringIO(structure_content))

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residue
                        residue_count += 1
                        for atom in residue:
                            if atom.name == 'CA':
                                plddt_values.append(atom.bfactor)
                                break

        avg_plddt = np.mean(plddt_values) if plddt_values else 0.0
        return avg_plddt, residue_count

    except Exception as e:
        print(f"    Warning: Error parsing PDB: {e}")
        # Fallback
        for line in structure_content.split('\n'):
            if line.startswith('ATOM'):
                try:
                    b_factor = float(line[60:66].strip())
                    plddt_values.append(b_factor)
                except:
                    pass
        avg_plddt = np.mean(plddt_values) if plddt_values else 0.0
        num_residues = structure_content.count('\nATOM') // 5
        return avg_plddt, num_residues


@app.route('/api/structure/<model_id>', methods=['GET'])
def get_structure(model_id):
    """Get structure data with coordinates and analysis"""
    try:
        # A model_id of "PDB-3NBY" means an experimentally solved structure
        # from the RCSB PDB; anything else is an AlphaFold prediction in the
        # usual "AF-P01308-F1-model_v4" format.
        source = 'experimental' if model_id.startswith('PDB-') else 'alphafold'
        # Optional: frontend can pass the UniProt accession explicitly (needed
        # for domain lookups on experimental structures, whose IDs don't encode it)
        uniprot_id = request.args.get('uniprot_id')
        version = None

        if source == 'experimental':
            pdb_id = model_id[len('PDB-'):]
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

            print(f"\n{'='*60}")
            print(f"Loading experimental structure for: {model_id}")
            print(f"  PDB ID: {pdb_id}")
            print(f"{'='*60}")

            print(f"Downloading from: {pdb_url}")
            response = requests.get(pdb_url, timeout=30)

            if response.status_code != 200:
                return jsonify({'error': f'Failed to download structure: {response.status_code}'}), 404

            pdb_content = response.text
            print(f"Downloaded {len(pdb_content)} bytes")

        else:
            # Extract UniProt ID from model_id
            # Format: AF-P01308-F1-model_v4 or AF-P01308-F1
            parts = model_id.split('-')
            uniprot_id = uniprot_id or parts[1]

            print(f"\n{'='*60}")
            print(f"Loading structure for: {model_id}")
            print(f"  UniProt ID: {uniprot_id}")
            print(f"{'='*60}")

            # Get structure info from API (returns list)
            entries = get_alphafold_structure_info(uniprot_id)

            if not entries:
                return jsonify({'error': 'Structure not found in AlphaFold database'}), 404

            # Find the matching entry (by model_id base)
            entry = None
            for e in entries:
                if model_id.startswith(e.get('entryId', '')):
                    entry = e
                    break

            # If no exact match, use first entry
            if not entry:
                entry = entries[0]

            # ALWAYS use the current 'pdbUrl' the AlphaFold API
            # just gave us for this entry, rather than parsing/trusting a
            # '-model_vN' suffix embedded in model_id (which used to default
            # to v4 whenever no suffix was present, and would silently 404
            # any time AlphaFold DB has since re-run and bumped that
            # protein's model to a newer version -- confirmed happening in
            # practice: AlphaFold DB is now on v6 for at least some entries,
            # while this code still assumed v4). The API's own pdbUrl is
            # already the correct, current file location -- no need to
            # reconstruct it, and reconstructing it is exactly what goes
            # stale.
            entry_id = entry.get('entryId', f"AF-{uniprot_id}-F1")
            version = entry.get('latestVersion')
            pdb_url = entry.get('pdbUrl') or f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v{version}.pdb"

            print(f"  Version (current, from API): {version}")
            print(f"Downloading from: {pdb_url}")

            response = requests.get(pdb_url, timeout=30)

            if response.status_code != 200:
                return jsonify({'error': f'Failed to download structure: {response.status_code}'}), 404

            pdb_content = response.text
            print(f"Downloaded {len(pdb_content)} bytes")

        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', StringIO(pdb_content))

        # Extract coordinates and properties
        coords = []
        plddt = []
        b_factors = []
        residue_names = []
        residue_numbers = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residue
                        for atom in residue:
                            if atom.name == 'CA':  # Alpha carbon
                                coords.append(atom.coord.tolist())
                                plddt.append(atom.bfactor)
                                b_factors.append(atom.bfactor)

                                try:
                                    res_name = three_to_one(residue.resname)
                                except:
                                    res_name = 'X'

                                residue_names.append(res_name)
                                residue_numbers.append(residue.id[1])

        print(f"Parsed {len(coords)} residues")

        # Calculate SASA (consensus RSA -- see calculate_sasa docstring).
        # Switched to return_stats=True so consensus_z/
        # agreement_sd (already computed internally either way) get exposed
        # to the frontend at initial load time, not just recomputed
        # separately inside unified_crm1_nes_analysis for NES candidates.
        # This lets /api/nls_scan build an equivalent rsa_profile for NLS
        # candidates too, without re-downloading/re-parsing the PDB itself
        # (it just receives these same whole-protein arrays from the
        # frontend and slices them, same as it already does for sasa/plddt).
        sasa_values, sasa_computed, consensus_z_values, agreement_sd_values = calculate_sasa(
            structure, pdb_text=pdb_content, return_stats=True)

        # Calculate hydrophobicity
        hydrophobicity = [HYDROPHOBICITY.get(res, 0.0) for res in residue_names]

        # Calculate charge
        charges = [CHARGE.get(res, 0.0) for res in residue_names]

        # Calculate disorder score
        disorder_scores = calculate_disorder_score(residue_names, plddt, sasa_values)

        # Calculate flexibility from pLDDT variance
        flexibility_scores = calculate_flexibility_from_plddt(plddt, window_size=5)

        # DON'T calculate CRM1/NES on initial load - too slow!
        # These will be calculated on-demand when user clicks CRM1 color mode
        sequence = ''.join(residue_names)

        print(f"Basic structure loaded ({len(coords)} residues)")

        # Fetch domain information from UniProt.
        # For AlphaFold models the UniProt ID is embedded in model_id
        # (format: AF-P12345-F1-model_v4). For experimental (PDB-XXXX)
        # structures it isn't, so we rely on the uniprot_id query param
        # the frontend passes along from the protein it was browsing.
        protein_id = None
        if source == 'alphafold' and model_id.startswith('AF-'):
            parts = model_id.split('-')
            if len(parts) >= 2:
                protein_id = parts[1]
        elif uniprot_id:
            protein_id = uniprot_id

        # IUPred2A/ANCHOR2: real sequence-based disorder + disordered-
        # binding-region prediction, now the PRIMARY disorder signal
        # wherever it's available -- replacing calculate_disorder_score()'s
        # pLDDT/composition heuristic, which is kept only as the fallback
        # (no UniProt mapping, network failure, unalignable sequence). This
        # matters especially for experimental structures, where 'plddt'
        # above is really a B-factor (see 'confidence_metric' below) that
        # the old heuristic was scoring as if it were disorder-relevant
        # AlphaFold confidence.
        structural_disorder_scores = disorder_scores  # old heuristic, kept for comparison/fallback
        iupred_aligned, anchor2_aligned = None, None
        disorder_source = 'structural_heuristic'
        if protein_id:
            iupred_raw, anchor2_raw, iupred_seq = fetch_iupred2a_scores(protein_id)
            iupred_aligned, anchor2_aligned = align_iupred_to_structure(
                sequence, iupred_seq, iupred_raw, anchor2_raw)
        if iupred_aligned:
            disorder_scores = iupred_aligned
            disorder_source = 'iupred2a'
        # else: disorder_scores stays as the calculate_disorder_score() output already computed above

        domains = []
        domain_colors_per_residue = ['#CCCCCC'] * len(residue_numbers)  # Default gray

        if protein_id:
            try:
                print(f"Fetching domain info for {protein_id}...")
                uniprot_url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
                domain_response = requests.get(uniprot_url, timeout=10)

                if domain_response.status_code == 200:
                    domain_data = domain_response.json()
                    features = domain_data.get('features', [])

                    domain_colors = [
                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
                        '#FF8C94', '#A8E6CF', '#FFD3B6', '#FFAAA5', '#C7CEEA'
                    ]
                    domain_idx = 0

                    for feature in features:
                        feature_type = feature.get('type', '')

                        if feature_type in ['Domain', 'Region', 'Repeat', 'Zinc finger', 'DNA binding']:
                            location = feature.get('location', {})
                            start = location.get('start', {}).get('value')
                            end = location.get('end', {}).get('value')

                            if start and end:
                                description = feature.get('description', 'Unknown domain')
                                color = domain_colors[domain_idx % len(domain_colors)]
                                domain_idx += 1

                                domains.append({
                                    'type': feature_type,
                                    'description': description,
                                    'start': start,
                                    'end': end,
                                    'color': color
                                })

                                # Color residues in this domain
                                for i, res_num in enumerate(residue_numbers):
                                    if start <= res_num <= end:
                                        domain_colors_per_residue[i] = color

                                print(f"  Domain: {description} ({start}-{end})")

                    print(f"  Total domains found: {len(domains)}")
            except Exception as domain_err:
                print(f"  Warning: Could not fetch domains: {domain_err}")

        # Extract x, y, z coordinates separately (frontend expects this format)
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        z_coords = [c[2] for c in coords]

        # Build sequence string
        sequence = ''.join(residue_names)

        # Real localCIDER linear hydropathy / NCPR / FCR profiles (one value
        # per residue) so they're available as structure color modes without
        # a second round-trip.
        cider_profiles = compute_linear_cider_profiles(sequence)

        structure_data = {
            'model_id': model_id,  # ADD THIS - needed for on-demand CRM1 analysis
            'source': source,  # 'alphafold' or 'experimental'
            # For AlphaFold models the per-residue value below is a genuine
            # pLDDT confidence (0-100). For experimental (PDB) structures the
            # PDB file's B-factor column is reused for the same array, but it
            # is a real crystallographic/cryo-EM B-factor, not a confidence
            # score -- 'confidence_metric' tells the frontend which is which.
            'confidence_metric': 'plddt' if source == 'alphafold' else 'bfactor',
            'coordinates': coords,
            'x': x_coords,
            'y': y_coords,
            'z': z_coords,
            'sequence': sequence,
            'plddt': plddt,
            'mean_plddt': round(np.mean(plddt), 2) if plddt else 0,  # Add mean pLDDT
            'bfactor': b_factors,  # B-factor (same as pLDDT in AlphaFold)
            'b_factors': b_factors,
            'residue_names': residue_names,
            'residue_numbers': residue_numbers,
            'hydrophobicity': hydrophobicity,
            'charges': charges,
            'charge': charges,  # Alternative name for consistency
            'sasa': sasa_values,
            'sasa_computed': sasa_computed,  # False = fallback placeholder, not real exposure data
            # Per-residue consensus z-score (exposure relative to the REST
            # of this protein) and cross-method agreement SD -- see
            # calculate_sasa(return_stats=True) / consensus_accessibility.py.
            # Sent up front now so any consumer (NLS scan, future panels)
            # can build a full RSA profile without a second structure fetch.
            'consensus_z': consensus_z_values,
            'agreement_sd': agreement_sd_values,
            'disorder': disorder_scores,  # Alternative name for consistency -- IUPred2A when available, else heuristic
            'disorder_scores': disorder_scores,
            'disorder_source': disorder_source,  # 'iupred2a' or 'structural_heuristic'
            'structural_disorder': structural_disorder_scores,  # old pLDDT/composition heuristic, always present
            'anchor2_binding': anchor2_aligned,  # disordered *binding* region probability (None if IUPred2A unavailable)
            'flexibility': flexibility_scores,  # Flexibility from pLDDT variance
            'domains': domains,
            'domain_colors': domain_colors_per_residue,
            'crm1_scores': None,  # Will be computed on-demand
            'nes_motifs': None,  # Will be computed on-demand
            'crm1_binding_regions': None,  # Will be computed on-demand
            'pdb_content': pdb_content,
            **cider_profiles,  # linear_hydropathy, linear_ncpr, linear_fcr, positions, cider_computed
        }

        print(f"Structure data prepared")
        print(f"{'='*60}\n")

        return jsonify(structure_data)

    except Exception as e:
        print(f"Error loading structure: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# NOTE: an orphaned `@app.route('/api/crm1_analysis_ondemand', methods=['POST'])`
# decorator used to sit directly on calculate_simple_disorder_score() below --
# meaning that helper function (used throughout this file, e.g. inside
# calculate_enhanced_disorder_score) was ALSO registered as this route's
# Flask view. Any POST to /api/crm1_analysis_ondemand would have crashed
# with "missing required positional argument: sequence" (Flask calls view
# functions with no args unless the route itself declares a <param>).
# Confirmed dead: the current frontend (App.jsx) never calls this URL -- the
# real on-demand CRM1/NES analysis path is GET /api/unified_crm1_nes/<model_id>
# (see App.jsx's handleColourModeChange). Only frontend/src/old files/App_separate.jsx
# (an unused legacy file) and old versions/app_separate.py reference this
# route name, from an earlier, since-superseded architecture. Decorator
# removed rather than "fixed" -- there's no live caller to build a working
# implementation against, and resurrecting the old app_separate.py version
# would reintroduce a second, inconsistent CRM1 scoring path (predating the
# consensus SASA/IUPred2A work) that nothing currently needs.

# ============================================================================
# UNIFIED CRM1/NES ANALYSIS ENDPOINT
# ============================================================================

def calculate_simple_disorder_score(sequence):
    """Calculate disorder propensity from sequence alone"""
    disorder_promoting = 'PREKSQDG'
    order_promoting = 'WYFIVLMC'

    disorder_count = sum(sequence.count(aa) for aa in disorder_promoting)
    order_count = sum(sequence.count(aa) for aa in order_promoting)

    if len(sequence) == 0:
        return 0.5

    score = (disorder_count - order_count) / len(sequence)
    return max(0, min(1, (score + 1) / 2))  # Normalize to 0-1


def fetch_uniprot_disorder_regions(uniprot_id):
    """
    Fetch disordered regions from UniProt annotations
    Returns list of dicts with start/end positions
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()
        disorder_regions = []

        features = data.get('features', [])
        for feature in features:
            feature_type = feature.get('type', '')

            # Look for disordered region annotations
            if feature_type in ['Region', 'Compositional bias']:
                description = feature.get('description', '').lower()
                if 'disorder' in description or 'disordered' in description or 'unstructured' in description:
                    location = feature.get('location', {})
                    start = location.get('start', {}).get('value')
                    end = location.get('end', {}).get('value')

                    if start and end:
                        disorder_regions.append({
                            'start': start,
                            'end': end,
                            'description': feature.get('description', 'Disordered region')
                        })
                        print(f"   Found UniProt disorder region: {start}-{end}")

        return disorder_regions

    except Exception as e:
        print(f"   Warning: Could not fetch UniProt disorder regions: {e}")
        return []


_UNIPROT_STRUCTURAL_ANNOTATIONS_CACHE = {}  # uniprot_id -> (coiled_coil_regions, domain_regions)

# UniProt curators don't consistently use the literal 'Coiled coil' feature
# type -- intermediate filament proteins (e.g. Lamin A/C) get 'Region:
# Coil 1A/1B/Coil 2'; bZIP transcription factors (e.g. JDP2) get 'Region:
# Leucine-zipper' nested inside a 'Domain: bZIP'. Matching on these
# description keywords too (not just the literal type) catches both --
# confirmed empirically (see fetch_uniprot_structural_annotations docstring)
# before this was wired into anything.
_COILED_COIL_DESC_KEYWORDS = ('coil', 'zipper')


def fetch_uniprot_structural_annotations(uniprot_id):
    """Real, curated/sequence-based UniProt feature annotations for telling
    a genuine coiled-coil/leucine-zipper region apart from an ordinary
    structured domain (e.g. bHLH, SH3-like) -- context a single AlphaFold
    monomer structure can't provide on its own.

    Added after _has_packed_second_helix() (real 3D-geometry
    "is there a second helix packed nearby") was tried and reverted: it came
    back False for nearly every genuine coiled-coil in the holdout set,
    because most real coiled-coils are DIMERS and AlphaFold DB only models
    one chain -- there's no partner in the structure to ever find, so
    "no second helix found" was never real evidence of anything. UniProt's
    own 'Coiled coil' annotations are usually computed straight from
    sequence (COILS/Marcoil-style heptad periodicity) or curated from the
    literature, so they don't depend on which chain got modeled at all.

    Verified against real UniProt data for known cases before being wired
    into any scoring: Myosin-9 (P35579) has a literal 'Coiled coil' feature
    at 837-1926, covering the 1067-1076 false-positive window dead center.
    Lamin A/C (P02545) has 'Region: Coil 2' at 243-383, covering the
    362-371 target window. TPM2 (P07951) has a literal 'Coiled coil' at
    1-284, covering its 4-13 target window. JDP2 (P97875, mouse) has
    'Region: Leucine-zipper' at 100-128, covering its 114-123 target
    window. Neurogenin-3 (Q9Y4Z2) has NO coiled-coil-type annotation at
    all -- only 'Domain: bHLH' at 83-135, which is real evidence AGAINST
    its 131-142 NES candidate being a coiled-coil, not for it.

    Returns (coiled_coil_regions, domain_regions), each a list of
    {'start', 'end', 'description'} dicts, cached per uniprot_id.
    domain_regions deliberately excludes anything already caught by the
    coil/zip keyword match above (so a 'Domain: bZIP' entry, itself a real
    leucine zipper, doesn't get miscounted as "evidence against" just
    because its type is 'Domain' rather than 'Coiled coil').

    NOT wired into scoring yet -- computed and available for candidates to
    check overlap against, same cautious "log first" rollout as
    _has_packed_second_helix originally had, until this is validated
    against a full live holdout run rather than just these 5 spot checks.
    """
    if uniprot_id in _UNIPROT_STRUCTURAL_ANNOTATIONS_CACHE:
        return _UNIPROT_STRUCTURAL_ANNOTATIONS_CACHE[uniprot_id]

    coiled_coil_regions = []
    domain_regions = []
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for feature in data.get('features', []):
                ftype = feature.get('type', '')
                desc = feature.get('description') or ''
                desc_lower = desc.lower()
                location = feature.get('location', {})
                start = location.get('start', {}).get('value')
                end = location.get('end', {}).get('value')
                if not start or not end:
                    continue

                is_coil_like = ftype == 'Coiled coil' or (
                    ftype in ('Region', 'Domain') and
                    any(kw in desc_lower for kw in _COILED_COIL_DESC_KEYWORDS)
                )
                if is_coil_like:
                    coiled_coil_regions.append({'start': start, 'end': end,
                                                 'description': desc or ftype})
                elif ftype in ('Domain', 'DNA binding'):
                    domain_regions.append({'start': start, 'end': end,
                                            'description': desc or ftype})

            print(f"   UniProt structural annotations for {uniprot_id}: "
                  f"{len(coiled_coil_regions)} coiled-coil/zipper region(s), "
                  f"{len(domain_regions)} other domain region(s)")
        else:
            print(f"   Warning: UniProt lookup for {uniprot_id} structural annotations "
                  f"returned HTTP {response.status_code}")
    except Exception as e:
        print(f"   Warning: Could not fetch UniProt structural annotations for {uniprot_id}: {e}")

    result = (coiled_coil_regions, domain_regions)
    _UNIPROT_STRUCTURAL_ANNOTATIONS_CACHE[uniprot_id] = result
    return result


_UNIPROT_LIPIDATION_ANNOTATIONS_CACHE = {}  # uniprot_id -> (lipid_sites, is_nonnuclear_anchored, location_text)

# Subcellular-location keywords that, on their OWN (with no 'Nucleus'
# anywhere in the same protein's location text), indicate real curated
# evidence this protein is membrane-anchored or secreted rather than a
# nuclear import cargo. Matched case-insensitively against UniProt's own
# curated SUBCELLULAR LOCATION comment text -- same "real annotation, not
# guessed from sequence" standard as _COILED_COIL_DESC_KEYWORDS above.
_NONNUCLEAR_ANCHOR_LOCATION_KEYWORDS = ('cell membrane', 'lipid-anchor', 'secreted', 'extracellular')


def fetch_uniprot_lipidation_annotations(uniprot_id):
    """Real, curated UniProt evidence for telling a genuine membrane-
    anchored/secreted cationic effector protein apart from a nuclear
    import cargo whose basic patch merely LOOKS similar -- the NLS-side
    analog of fetch_uniprot_structural_annotations() above (same "explicit
    override from a real curated annotation, not a learned proxy" pattern
    that already fixed the coiled-coil false positives on the NES side and
    the CAAX-box false positives on the NLS side).

    Added after the 25+25 holdout diagnosis showed 3 of the
    remaining NLS false positives (MARCKS, GAP-43/neuromodulin, and
    Cathelicidin/LL-37) are membrane-binding or secreted cationic effector
    domains, not nuclear cargo -- confirmed directly against real UniProt
    data before wiring this into anything: MARCKS (P29966) has a curated
    'Lipidation: N-myristoyl glycine' at Gly-2 and 'SUBCELLULAR LOCATION:
    Cell membrane; Lipid-anchor'. GAP-43 (P17677) has 'Lipidation:
    S-palmitoyl cysteine' at Cys-3/Cys-4 and 'Cell membrane; Peripheral
    membrane protein'. Cathelicidin/CAMP (P49913) has no lipidation site
    but is curated 'Secreted; Vesicle' -- a neutrophil-granule/skin
    antimicrobial peptide, not lipidated but still definitively
    non-nuclear. None of the three has 'Nucleus' anywhere in their
    location text.

    A per-residue proximity check (like the CAAX veto's C-terminal
    distance) doesn't work here -- MARCKS's myristoylation site is at
    residue 2 but its false-positive window is 150 residues away, because
    a single N-terminal lipid anchor tethers the WHOLE protein to a
    membrane regardless of where in the sequence a basic patch happens to
    sit. So this is a protein-level veto: fires only when there's real
    lipidation/membrane/secreted evidence AND no 'Nucleus' anywhere in the
    same protein's curated location text (guards against vetoing genuine
    dual nucleus/membrane shuttling proteins, which do exist and would
    have 'Nucleus' alongside the membrane annotation).

    Returns (lipid_sites, comment_texts, location_text, mature_chain_ranges,
    cleaved_ranges):
      lipid_sites -- list of {'position', 'description'} dicts from real
        'Lipidation' features.
      comment_texts -- list of strings, one per curated 'SUBCELLULAR
        LOCATION' comment block (i.e. one per isoform when UniProt curates
        isoforms separately). Kept un-pooled so callers can reason about
        each isoform's own annotation instead of one merged blob -- see
        nonnuclear_anchor_factor() for why that distinction matters.
      location_text -- all comment_texts joined, for display/audit only.
      mature_chain_ranges -- list of (start, end) 1-indexed inclusive
        UniProt residue ranges for this protein's innermost/most-processed
        'Chain'-typed features (see nonnuclear_anchor_factor() docstring).
        Empty if no maturation complexity is annotated.
      cleaved_ranges -- list of (start, end) for 'Signal'/'Propeptide'/
        'Transit peptide'-typed features -- residues that are always
        removed before ANY final localized form exists, regardless of
        isoform.
    Actual veto decision (is_nonnuclear_anchored / anchor_factor) is
    computed per-candidate-window by nonnuclear_anchor_factor() below, not
    here -- see that function's docstring for the full history of why this
    moved from a pure protein-level cached decision to a window-aware one.
    Raw UniProt data fetch is still cached per uniprot_id.
    """
    if uniprot_id in _UNIPROT_LIPIDATION_ANNOTATIONS_CACHE:
        return _UNIPROT_LIPIDATION_ANNOTATIONS_CACHE[uniprot_id]

    lipid_sites = []
    comment_texts = []
    location_text = ''
    mature_chain_ranges = []
    cleaved_ranges = []
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            chain_ranges = []
            for feature in data.get('features', []):
                ftype = feature.get('type')
                if ftype == 'Lipidation':
                    location = feature.get('location', {})
                    pos = location.get('start', {}).get('value')
                    if pos:
                        lipid_sites.append({'position': pos,
                                             'description': feature.get('description') or 'Lipidation'})
                elif ftype in ('Chain', 'Signal', 'Propeptide', 'Transit peptide'):
                    loc = feature.get('location', {})
                    s = loc.get('start', {}).get('value')
                    e = loc.get('end', {}).get('value')
                    if s is None or e is None:
                        continue
                    if ftype == 'Chain':
                        chain_ranges.append((s, e))
                    else:
                        cleaved_ranges.append((s, e))
            # Innermost/most-processed Chain ranges only -- UniProt curates
            # precursor -> mature processing as a series of Chain features
            # that shrink toward the final product (see
            # nonnuclear_anchor_factor() docstring for the real ORF2/
            # Lactoferrin example this is built from), so a Chain that
            # strictly contains another Chain is a precursor state, not the
            # final one.
            for i, (s, e) in enumerate(chain_ranges):
                if not any(j != i and s <= s2 and e2 <= e and (s, e) != (s2, e2)
                           for j, (s2, e2) in enumerate(chain_ranges)):
                    mature_chain_ranges.append((s, e))

            for comment in data.get('comments', []):
                if comment.get('commentType') != 'SUBCELLULAR LOCATION':
                    continue
                block_strings = []
                for loc in comment.get('subcellularLocations', []):
                    val = (loc.get('location') or {}).get('value')
                    if val:
                        block_strings.append(val)
                    topo = (loc.get('topology') or {}).get('value')
                    if topo:
                        block_strings.append(topo)
                if block_strings:
                    comment_texts.append('; '.join(block_strings))
            location_text = '; '.join(comment_texts)

            print(f"   UniProt lipidation/location annotations for {uniprot_id}: "
                  f"{len(lipid_sites)} lipidation site(s), location='{location_text}', "
                  f"mature_chain_ranges={mature_chain_ranges}, cleaved_ranges={cleaved_ranges}")
        else:
            print(f"   Warning: UniProt lookup for {uniprot_id} lipidation annotations "
                  f"returned HTTP {response.status_code}")
    except Exception as e:
        print(f"   Warning: Could not fetch UniProt lipidation annotations for {uniprot_id}: {e}")

    result = (lipid_sites, comment_texts, location_text, mature_chain_ranges, cleaved_ranges)
    _UNIPROT_LIPIDATION_ANNOTATIONS_CACHE[uniprot_id] = result
    return result


def _range_overlap_frac(candidate_start, candidate_end, ranges_1idx):
    """Best single-range overlap fraction of 0-indexed inclusive
    [candidate_start, candidate_end] against a list of 1-indexed inclusive
    UniProt ranges -- same convention/limitation (best single range, not a
    true union) as dna_binding_domain_factor() elsewhere in this file."""
    cand_len = candidate_end - candidate_start + 1
    if cand_len <= 0 or not ranges_1idx:
        return 0.0
    best = 0.0
    for s, e in ranges_1idx:
        s0, e0 = s - 1, e - 1
        overlap = min(candidate_end, e0) - max(candidate_start, s0) + 1
        if overlap > 0:
            best = max(best, overlap / cand_len)
    return best


# Same overlap-fraction convention as NLS_DNA_BINDING_DOMAIN_MIN_OVERLAP_FRAC.
NLS_MATURE_CHAIN_MIN_OVERLAP_FRAC = 0.5


def nonnuclear_anchor_factor(candidate_start, candidate_end, lipid_sites, comment_texts,
                              mature_chain_ranges, cleaved_ranges):
    """Per-candidate-window anchor/secreted veto. Returns (is_nonnuclear_
    anchored, anchor_factor, used_lipid_evidence).

    history, in order:
    v1/v2: single protein-level decision, POOLING every 'SUBCELLULAR
    LOCATION' comment's text together before checking for anchor keywords
    / 'Nucleus'. Missed Lactoferrin (P02788, 0.556 false positive) --
    UniProt curates it with TWO separate comments, one for the canonical
    secreted isoform ('Secreted; Cytoplasmic granule') and one for a
    distinct alternative isoform 'DeltaLf' ('Cytoplasm; Nucleus'). Pooling
    let DeltaLf's Nucleus mention suppress the veto for the canonical
    isoform too, even though the flagged window has nothing to do with
    DeltaLf.
    v3 (tried, reverted): switched to a PER-BLOCK check -- a block only
    counts if it itself has an anchor keyword and doesn't itself mention
    Nucleus. Fixed Lactoferrin, but broke Hepatitis E ORF2 capsid protein
    (P29326, was a solid 0.929 true positive): ORF2 has the identical
    two-comment shape (one 'Secreted' block, one '...Host nucleus' block),
    but the real literature NLS at residues 28-33 belongs to the isoform
    the NUCLEUS block describes, and per-block matching fired off the
    unrelated 'Secreted' block instead.
    v4 (this version): per-block text check, PLUS a Chain-boundary
    plausibility check on top when UniProt has annotated processing
    complexity. The actual distinguishing fact between the two cases isn't
    isoform naming (Lactoferrin's canonical isoform is just named "1",
    nothing to fuzzy-match against a Chain description) -- it's whether the
    candidate window survives into an annotated MATURE chain at all.
    Lactoferrin's flagged window (46-51) sits inside its single mature
    'Lactotransferrin' chain (20-710) -- genuinely part of the secreted
    molecule. ORF2's flagged window (28-33) sits inside the broader,
    IMMATURE 'Pro-secreted protein ORF2' chain (20-660) but outside the
    narrower, mature 'Secreted protein ORF2' chain (34-660) nested inside
    it -- i.e. that window is part of a propeptide segment clipped off
    before the protein ever becomes 'Secreted protein ORF2', so the
    'Secreted' block's evidence shouldn't apply to it. Using the innermost/
    most-processed Chain range as the reference (see
    fetch_uniprot_lipidation_annotations()'s mature_chain_ranges) resolves
    both correctly without any isoform-name matching. When a protein has NO
    Chain-typed features at all (MARCKS, GAP-43, Cathelicidin -- none of
    the three original calibration cases have any), this check is skipped
    entirely and the per-block text evidence is trusted as-is, so those
    three are unaffected.
    """
    location_lower_all = ' '.join(comment_texts).lower()
    has_nucleus_anywhere = 'nucleus' in location_lower_all or 'nucleolus' in location_lower_all
    lipid_evidence = bool(lipid_sites) and not has_nucleus_anywhere

    location_evidence = False
    for txt in comment_texts:
        txt_lower = txt.lower()
        if not any(kw in txt_lower for kw in _NONNUCLEAR_ANCHOR_LOCATION_KEYWORDS):
            continue
        if 'nucleus' in txt_lower or 'nucleolus' in txt_lower:
            continue
        if mature_chain_ranges:
            if _range_overlap_frac(candidate_start, candidate_end,
                                    mature_chain_ranges) < NLS_MATURE_CHAIN_MIN_OVERLAP_FRAC:
                continue  # window doesn't survive into any mature chain -- skip this block
            if _range_overlap_frac(candidate_start, candidate_end,
                                    cleaved_ranges) >= NLS_MATURE_CHAIN_MIN_OVERLAP_FRAC:
                continue  # window is mostly a signal/propeptide segment -- skip this block
        location_evidence = True
        break

    is_nonnuclear_anchored = lipid_evidence or location_evidence
    anchor_factor = 1.0
    if is_nonnuclear_anchored:
        anchor_factor = (NLS_NONNUCLEAR_ANCHOR_FACTOR_LIPID if lipid_evidence
                          else NLS_NONNUCLEAR_ANCHOR_FACTOR_LOCATION_ONLY)
    return is_nonnuclear_anchored, anchor_factor, lipid_evidence


# NLS_NONNUCLEAR_ANCHOR_PROBABILITY_CAP kept as the name/
# value for the strong-evidence (real lipidation site) tier, for continuity
# with existing callers/docs; NLS_NONNUCLEAR_ANCHOR_FACTOR_LIPID is the same
# constant under its new, more accurate name. Cap rather than hard-zero,
# same rationale as CAAX_PROBABILITY_CAP -- a strong biological prior (dual
# nucleus/membrane shuttling proteins with incomplete UniProt annotation
# are possible), not logical certainty.
NLS_NONNUCLEAR_ANCHOR_PROBABILITY_CAP = 0.15
NLS_NONNUCLEAR_ANCHOR_FACTOR_LIPID = NLS_NONNUCLEAR_ANCHOR_PROBABILITY_CAP
# Location-text-only evidence (no covalent lipidation site) is real curated
# UniProt data but weaker/less specific than a named lipidation residue --
# see anchor_factor docstring above. Calibrated against the real holdout
# case it exists for (Cathelicidin/LL-37, P49913: 'Secreted; Vesicle', no
# Lipidation feature) below.
NLS_NONNUCLEAR_ANCHOR_FACTOR_LOCATION_ONLY = 0.35


_UNIPROT_DNA_BINDING_ANNOTATIONS_CACHE = {}  # uniprot_id -> [{'start','end','description'},...]


def fetch_uniprot_dna_binding_regions(uniprot_id):
    """Real, curated UniProt 'DNA binding' feature-type regions -- the NLS
    side's analog of the coiled-coil check, for the hardest documented NLS
    failure mode (see nls_ml_predictor.py's module docstring: "NLS tools
    over-fire on ANY K/R-rich stretch... this model's hard negatives are
    real UniProt-annotated DNA-binding regions"). This project's own
    training data already has 102 such curated hard negatives
    (neg_type='dna_binding_hard') and the classifier still missed one on
    a real holdout run: Engrailed homeodomain (P02836), nls_probability
    0.684, target window 454-513.

    Deliberately scoped to UniProt's precise 'DNA binding' feature TYPE
    only (not the generic 'Domain'/'Region' types fetch_uniprot_
    structural_annotations() also collects) -- checked directly against 4
    real DNA-binding transcription-factor holdout POSITIVES before being
    wired into anything, specifically because a real NLS often sits near
    or inside a real DNA-binding-associated region and a broad match would
    risk vetoing genuine cargo:
      - p53 (P04637): 'DNA binding' feature at 102-292; its real NLS target
        window is 305-321 -- no overlap.
      - NF-kB p50/NFKB1 (P19838): NO 'DNA binding'-typed feature at all --
        its DNA-binding+dimerization+NLS-containing Rel-homology domain is
        curated as type 'Domain' ("RHD", 42-367, which DOES cover its
        360-365 target window), not 'DNA binding'. This is exactly why the
        type filter is strict: a keyword/generic-domain match here would
        have vetoed a real positive.
      - c-Myc (P01106): 'Domain: bHLH' at 369-421 (not 'DNA binding'
        typed); its real NLS target window is 335-343 -- no overlap either
        way.
      - RB1 (P06400): no 'DNA binding'-typed feature at all.
      - Engrailed (P02836): 'DNA binding' feature at 454-513 ("Homeobox"),
        which is exactly its false-positive target window.
    Only checked against these 4 positives + the 1 target case, NOT the
    full 254-example real training positive set the other two vetoes in
    this file were calibrated against -- that would need a live UniProt
    fetch per protein and wasn't feasible to run in bulk here. Treat
    dna_binding_domain_factor as provisional until validated against a
    full live holdout run.

    Returns a list of {'start', 'end', 'description'} dicts (1-indexed,
    UniProt numbering), cached per uniprot_id.
    """
    if uniprot_id in _UNIPROT_DNA_BINDING_ANNOTATIONS_CACHE:
        return _UNIPROT_DNA_BINDING_ANNOTATIONS_CACHE[uniprot_id]

    regions = []
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=accession,ft_dna_bind"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for feature in data.get('features', []):
                if feature.get('type') != 'DNA binding':
                    continue
                location = feature.get('location', {})
                start = location.get('start', {}).get('value')
                end = location.get('end', {}).get('value')
                if start and end:
                    regions.append({'start': start, 'end': end,
                                     'description': feature.get('description') or 'DNA binding'})
            print(f"   UniProt DNA-binding-region annotations for {uniprot_id}: {len(regions)} region(s)")
        else:
            print(f"   Warning: UniProt lookup for {uniprot_id} DNA-binding annotations "
                  f"returned HTTP {response.status_code}")
    except Exception as e:
        print(f"   Warning: Could not fetch UniProt DNA-binding annotations for {uniprot_id}: {e}")

    _UNIPROT_DNA_BINDING_ANNOTATIONS_CACHE[uniprot_id] = regions
    return regions


# Continuous ramp on the FRACTION of the candidate window covered by a real
# DNA-binding-typed region -- same "graded by strength of real evidence"
# principle as the other two vetoes above, not a binary cliff.
#
# A 15-residue FLANK on the region boundaries (tried to
# fix Engrailed homeodomain, P02836) was reverted after a real holdout
# regression -- SV40 Large T-antigen (P03070, PPKKKRKV, the field's most
# canonical monopartite NLS) got newly false-veto'd. Root cause, confirmed
# with real UniProt boundary data via check_dna_binding_boundaries.py:
# SV40's real NLS (125-132) and its DNA-binding origin-binding domain
# (139-254) have a genuine, real 7-residue GAP between them -- zero direct
# overlap. Flanking bridges gaps like that into fake overlap, which is
# exactly backwards: a real gap is evidence the NLS and the DBD are
# different features, not the same one.
#
# Engrailed's case is different in kind, not just degree: its flagged
# candidate (438-458) already has REAL, DIRECT, unflanked overlap with its
# Homeobox annotation (454-513) -- 5 of 21 residues, ~24% -- it's just a
# small fraction of a longer candidate window, not a gap. So instead of
# flanking (which affects gap cases like SV40/p53 too), MIN_OVERLAP_FRAC is
# lowered and a separate saturation point makes the ramp reach the floor
# discount much faster once genuine overlap exists, rather than needing
# 100% coverage. Critically this can NEVER discount a zero-overlap case:
# SV40 (0% direct overlap) and p53 (0% direct overlap, real NLS 305-321 vs
# DNA_BIND 102-292, 13-residue gap) stay at exactly best_overlap_frac=0.0,
# below any positive threshold, completely unaffected regardless of how
# steep the ramp is beyond it -- structurally safer than flanking, which
# could turn any small enough gap into fake overlap. Verified by hand:
# Engrailed's 24% overlap -> factor ~0.45 (0.999*0.45=0.45, clears the 0.5
# cutoff); SV40/p53 stay at factor 1.0 (unchanged, 0% overlap never reaches
# MIN_OVERLAP_FRAC). NF-kB p50/c-Myc/RB1 still have no 'DNA binding'-typed
# feature at all, so they're unaffected regardless.
#
# Still: only checked against these calibration cases + Engrailed, not the
# full holdout set -- re-run the full holdout test and confirm sensitivity
# didn't drop before trusting this (same lesson as the flank attempt).
NLS_DNA_BINDING_DOMAIN_MIN_OVERLAP_FRAC = 0.10
NLS_DNA_BINDING_DOMAIN_RAMP_SATURATION_FRAC = 0.30  # floor discount reached by this much REAL overlap, not 100%
NLS_DNA_BINDING_DOMAIN_FACTOR_FLOOR = 0.20


def dna_binding_domain_factor(candidate_start, candidate_end, dna_binding_regions):
    """Multiplicative discount in [NLS_DNA_BINDING_DOMAIN_FACTOR_FLOOR, 1.0],
    ramped by what fraction of [candidate_start, candidate_end] (0-indexed
    start, inclusive end -- same convention as scan_sequence candidates) is
    DIRECTLY covered (no flanking -- see the constants' comment above for
    why) by any real UniProt 'DNA binding' region (1-indexed, inclusive,
    from fetch_uniprot_dna_binding_regions()). Returns 1.0 if there's no
    real annotation or the overlap doesn't clear the confidence floor."""
    if not dna_binding_regions:
        return 1.0
    cand_len = candidate_end - candidate_start + 1
    if cand_len <= 0:
        return 1.0
    best_overlap_frac = 0.0
    for region in dna_binding_regions:
        r_start, r_end = region['start'] - 1, region['end'] - 1  # -> 0-indexed, NOT flanked
        overlap = min(candidate_end, r_end) - max(candidate_start, r_start) + 1
        if overlap > 0:
            best_overlap_frac = max(best_overlap_frac, overlap / cand_len)
    if best_overlap_frac < NLS_DNA_BINDING_DOMAIN_MIN_OVERLAP_FRAC:
        return 1.0
    if best_overlap_frac >= NLS_DNA_BINDING_DOMAIN_RAMP_SATURATION_FRAC:
        return NLS_DNA_BINDING_DOMAIN_FACTOR_FLOOR
    span = NLS_DNA_BINDING_DOMAIN_RAMP_SATURATION_FRAC - NLS_DNA_BINDING_DOMAIN_MIN_OVERLAP_FRAC
    ramp = (best_overlap_frac - NLS_DNA_BINDING_DOMAIN_MIN_OVERLAP_FRAC) / span
    return 1.0 - ramp * (1.0 - NLS_DNA_BINDING_DOMAIN_FACTOR_FLOOR)


def _overlaps_any_region(start, end, regions):
    """True if [start, end] overlaps any {'start','end',...} dict in
    `regions` (both inclusive, real residue numbering -- same convention
    used throughout this route for target_window/candidate overlap
    checks)."""
    return any(start <= r['end'] and r['start'] <= end for r in regions)


# =============================================================================
# IUPRED2A / ANCHOR2 -- real sequence-based disorder + disordered-binding-
# region prediction, replacing the old pLDDT/composition heuristic as the
# PRIMARY disorder signal wherever it's available.
#
# Why: pLDDT is only a genuine disorder proxy for AlphaFold models. For
# experimental (PDB-XXXX) structures, /api/structure's 'plddt' array is
# actually the crystallographic/cryo-EM B-factor column reused for the same
# slot (see 'confidence_metric' there) -- not comparable to AlphaFold
# confidence, so scoring it as if low-B-factor meant "disordered" was never
# valid. IUPred2A is sequence-only, so it's equally valid regardless of
# structure source. pLDDT is NOT removed anywhere -- it's still returned in
# full and still used for structural confidence/flexibility -- its weight is
# just reduced specifically in the places it was standing in for disorder.
#
# ANCHOR2 (predicted disordered *binding* regions -- segments that fold up
# on engaging a partner) is fetched in the same request. NES/NLS motifs are
# themselves short linear binding elements inside disordered regions (a NES
# engages CRM1's groove, a classical NLS engages importin-alpha), which is
# exactly what ANCHOR2 is designed to flag -- so it's wired in as a targeted
# bonus alongside general disorder, not just displayed.
# =============================================================================
_IUPRED2A_CACHE = {}  # uniprot_id -> (iupred_scores, anchor2_scores, iupred_sequence) or None


def fetch_iupred2a_scores(uniprot_id):
    """
    Fetch per-residue IUPred2A disorder scores + ANCHOR2 binding-region
    scores for a UniProt accession, via IUPred2A's hosted REST API (one
    request, type=anchor, returns both arrays -- verified against a live
    call: https://iupred2a.elte.hu/iupred2a/anchor/{accession}.json ->
    {"sequence": ..., "iupred2": [...], "anchor2": [...]}).

    Returns (iupred_scores, anchor2_scores, sequence) -- all None on any
    failure (bad accession, network error, non-200, malformed JSON) so
    callers can fall back to the existing heuristic exactly like
    fetch_uniprot_disorder_regions()'s callers already do for [].

    Cached in-memory per accession for the life of the process -- neither
    this nor fetch_uniprot_disorder_regions() cached before, so repeated
    analysis of the same protein was re-hitting both external services
    every time.
    """
    if not uniprot_id:
        return None, None, None

    if uniprot_id in _IUPRED2A_CACHE:
        return _IUPRED2A_CACHE[uniprot_id]

    result = (None, None, None)
    try:
        url = f"https://iupred2a.elte.hu/iupred2a/anchor/{uniprot_id}.json"
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print(f"   Warning: IUPred2A returned HTTP {response.status_code} for {uniprot_id}")
        else:
            data = response.json()
            iupred_scores = data.get('iupred2')
            anchor2_scores = data.get('anchor2')
            iupred_sequence = data.get('sequence')

            if iupred_scores and iupred_sequence:
                print(f"   IUPred2A: fetched {len(iupred_scores)}-residue disorder profile for {uniprot_id}"
                      + (f" (+ANCHOR2)" if anchor2_scores else " (no ANCHOR2 in response)"))
                result = (iupred_scores, anchor2_scores, iupred_sequence)
            else:
                print(f"   Warning: IUPred2A response for {uniprot_id} missing expected fields")

    except Exception as e:
        print(f"   Warning: Could not fetch IUPred2A scores for {uniprot_id}: {e}")

    _IUPRED2A_CACHE[uniprot_id] = result
    return result


def align_iupred_to_structure(structure_sequence, iupred_sequence, iupred_scores, anchor2_scores):
    """
    IUPred2A returns scores for the FULL UniProt canonical sequence, but the
    structure being analyzed may cover only part of it (AlphaFold splits
    proteins over ~2700 residues into multiple F1/F2/F3 fragments; PDB
    structures may be missing residues at the termini). This aligns the
    IUPred/ANCHOR2 arrays to structure_sequence's indexing before anything
    downstream tries to slice them by structure-derived array index.

    Returns (aligned_iupred, aligned_anchor2) -- both None if alignment
    can't be established, so callers fall back to the existing heuristic
    rather than silently scoring residues against the wrong positions.
    """
    if not iupred_scores or not iupred_sequence or not structure_sequence:
        return None, None

    # Exact match (by far the common case -- single-fragment AlphaFold
    # models and most PDB entries cover the full canonical sequence).
    if iupred_sequence == structure_sequence:
        return iupred_scores, anchor2_scores

    # Structure is a substring of the canonical sequence (fragment offset,
    # or missing terminal residues in a PDB structure).
    offset = iupred_sequence.find(structure_sequence)
    if offset != -1:
        end = offset + len(structure_sequence)
        aligned_anchor2 = anchor2_scores[offset:end] if anchor2_scores else None
        return iupred_scores[offset:end], aligned_anchor2

    # Give up rather than guess -- misaligned disorder scores are worse
    # than no IUPred data (caller falls back to the structural heuristic).
    print(f"   Warning: could not align IUPred2A sequence ({len(iupred_sequence)} aa) to "
          f"structure sequence ({len(structure_sequence)} aa) -- falling back to heuristic disorder")
    return None, None


def calculate_enhanced_disorder_score(sequence, start_pos, end_pos, plddt_values, uniprot_disorder_regions,
                                       iupred_fragment=None, anchor2_fragment=None):
    """
    Enhanced disorder calculation. PRIMARY signal is now IUPred2A
    (`iupred_fragment`, a per-residue array pre-sliced by the caller to the
    same window as `sequence`/`plddt_values` -- see align_iupred_to_structure)
    when it was available for this protein. The old literature-propensity
    calculate_simple_disorder_score() is now only the FALLBACK for when
    IUPred2A couldn't be fetched (no UniProt mapping, network failure,
    unalignable sequence).

    pLDDT is kept as a secondary signal, not removed -- but its weight here
    is HALVED (40% -> 20%) now that a real disorder predictor is available.
    It was never a fully trustworthy disorder proxy to begin with: it's
    literally a B-factor, not a confidence score, for experimental (PDB)
    structures (see /api/structure's 'confidence_metric' field), and even
    for genuine AlphaFold pLDDT it's a structure-CONFIDENCE metric being
    used as a disorder proxy, not a direct disorder measurement.

    ANCHOR2 (`anchor2_fragment`, disordered *binding* region probability,
    fetched in the same IUPred2A call as iupred_fragment) is folded in as a
    targeted bonus -- NES/NLS motifs are themselves short linear binding
    elements in disordered regions, which is exactly what ANCHOR2 predicts.
    """
    have_iupred = iupred_fragment is not None and len(iupred_fragment) > 0

    if have_iupred:
        primary_disorder = float(np.mean(iupred_fragment))
        disorder_source = 'iupred2a'
    else:
        primary_disorder = calculate_simple_disorder_score(sequence)
        disorder_source = 'sequence_propensity'

    # pLDDT-based flexibility (secondary signal -- see docstring above)
    if plddt_values and len(plddt_values) > 0:
        avg_plddt = np.mean(plddt_values)
        # pLDDT < 50 = very disordered, pLDDT > 90 = very ordered
        plddt_disorder = max(0, min(1, (90 - avg_plddt) / 40))
    else:
        plddt_disorder = 0.5

    # UniProt disorder annotation (binary: is it in a disordered region?)
    in_uniprot_disorder = False
    for disorder_region in uniprot_disorder_regions:
        disorder_start = disorder_region.get('start', 0)
        disorder_end = disorder_region.get('end', 0)
        if not (end_pos < disorder_start or start_pos > disorder_end):
            # Overlaps with disorder region
            in_uniprot_disorder = True
            break

    uniprot_disorder_bonus = 0.3 if in_uniprot_disorder else 0.0

    # ANCHOR2 disordered-binding-region bonus (see docstring)
    if anchor2_fragment is not None and len(anchor2_fragment) > 0:
        anchor2_score = float(np.mean(anchor2_fragment))
    else:
        anchor2_score = 0.0

    if have_iupred:
        # IUPred2A primary (50%), pLDDT secondary (20%, down from 40%),
        # UniProt annotation bonus unchanged (30%), ANCHOR2 bonus (15%) new.
        combined_disorder = (primary_disorder * 0.5 + plddt_disorder * 0.2 +
                              uniprot_disorder_bonus + anchor2_score * 0.15)
    else:
        # No IUPred2A available for this protein -- original weighting
        # (sequence-propensity 30% / pLDDT 40% / UniProt bonus 30%).
        combined_disorder = (primary_disorder * 0.3 + plddt_disorder * 0.4 + uniprot_disorder_bonus)

    combined_disorder = min(1.0, combined_disorder)

    return combined_disorder, {
        'sequence_disorder': round(primary_disorder, 3),  # key name kept for backward compat
        'disorder_source': disorder_source,
        'plddt_flexibility': round(plddt_disorder, 3),
        'in_uniprot_disorder_region': in_uniprot_disorder,
        'anchor2_binding': round(anchor2_score, 3),
        'combined': round(combined_disorder, 3)
    }


# ============================================================================
# IMPROVED NES FILTERING FUNCTIONS
# ============================================================================

def validate_nes_leucine_requirement(sequence):
    """
    HYDROPHOBIC + SPACING REQUIREMENT (restoration; leucine-gate
    removed -- see below. Name kept as-is for backward
    compatibility with existing callers/keys (calculate_improved_nes_score,
    components['leucine_filter']) even though it no longer requires a
    literal leucine.).

    The original design only ever generated candidates by matching one of
    the spaced NES consensus patterns (Phi-x(2,3)-Phi-x(2,3)-Phi-x-Phi and
    relatives) in the first place -- spacing was enforced BY CONSTRUCTION,
    not checked separately. That got lost when this became a standalone
    count/frequency-only gate: a window could have plenty of hydrophobics
    scattered in the wrong positions, clear a pure count/frequency check,
    score reasonably on the ML model's continuous features (pssm_score,
    frac_phi_total, mean_hydro -- none of which require an exact match),
    and still never match any of the six discrete consensus patterns --
    landing in Class: unknown even as a top hit. Restored here: a candidate
    must match at least one spaced consensus pattern (check_consensus_patterns,
    strict OR relaxed) AND have >= 4 hydrophobic (LIVMF) residues, matching
    the original version's requirement (both this docstring's own summary,
    above, and check_consensus_patterns' own 'relaxed' patterns already
    only require the fuller Phi={L,I,V,F,M} vocabulary -- see that
    function). ml_score keeps its dominant weight (0.78 in
    NES_HEURISTIC_WEIGHTS) in the final combined score -- this only changes
    which candidates are considered at all, not how much the ML model's
    verdict counts once a candidate passes.

    this function used to ALSO hard-reject any candidate with
    zero literal leucines ("Must have at least 1 leucine"), on top of the
    two checks the docstring above actually describes -- a third,
    undocumented gate that isn't part of "the original version's
    requirement" this function is supposed to restore. That's inconsistent
    with Kosugi et al. 2008 (Traffic), who screened 101 real CRM1-dependent
    NESs and found hydrophobics other than leucine were "typically allowed
    at all positions" (Phi = L, V, I, F, or M) -- a genuinely leucine-free
    NES built entirely from Ile/Val/Met/Phe is exactly the kind of sequence
    their expanded consensus predicts should exist and bind CRM1, and
    check_consensus_patterns' relaxed patterns (r'[LIVFM]...') already
    accept it. The absolute leucine requirement below was rejecting that
    whole class of real, literature-supported NES before it ever reached
    scoring -- removed; the hydrophobic-count and consensus-pattern checks
    (which do NOT require a literal leucine, only the Phi vocabulary) are
    the real gates now, consistent with both this docstring and Kosugi 2008.
    """
    hydrophobic = sum(sequence.count(aa) for aa in 'LIVMF')

    if hydrophobic < 4:
        return False, f"Insufficient hydrophobics (LIVMF): {hydrophobic} (need ≥4)"

    has_pattern, strength, pattern_msg = check_consensus_patterns(sequence)
    if not has_pattern:
        return False, f"No spaced consensus pattern matched ({pattern_msg})"

    return True, f"Valid: {hydrophobic} hydrophobics, spacing={strength} ({pattern_msg})"


def check_consensus_patterns(sequence):
    """STRICT CONSENSUS PATTERN MATCHING with leucine preference

    every pattern below used to end in a bare `.` (exactly one
    residue) for the FINAL Phi3-to-Phi4 gap, even though the first two gaps
    were already given the realistic {2} / {3} range Kosugi et al. 2008
    (Traffic) observed across their 101 real CRM1-dependent NESs. That
    asymmetry was never justified by the underlying biology -- Kosugi's
    spacer-length variability applies to all the junctions between anchors,
    not just the first two -- and it was undercounted here purely by
    historical accident. It surfaced as a real, concrete miss: the
    experimentally-validated SARS-CoV-2 nucleocapsid NES
    (ALALLLLDRLNQLESKMSG, Sendino/Omaetxebarria/Rodriguez 2020, bioRxiv
    2020.10.06.328138) has a 3-residue final gap and matched NONE of the
    13 patterns below, so it never reached ML/CRM1 scoring at all -- caught
    via a holdout test against real, independently-sourced viral NES data.
    Fix: widen the final gap to {1,3}, matching the range already used for
    the first two gaps, so multi-residue final spacers (like this real NES)
    are no longer structurally unmatchable.
    """
    # Strict patterns (prefer L at key positions)
    patterns_strict = [
        r'L.{2}[LIV].{2}[LIV].{1,3}[LIV]',
        r'[LIV].{2}L.{2}[LIV].{1,3}[LIV]',
        r'[LIV].{2}[LIV].{2}L.{1,3}[LIV]',
        r'L.{3}[LIV].{2}[LIV].{1,3}[LIV]',
        r'[LIV].{3}L.{2}[LIV].{1,3}[LIV]',
        r'L.{2}[LIV].{3}[LIV].{1,3}[LIV]',
        r'[LIV].{2}L.{3}[LIV].{1,3}[LIV]',
        r'L.{3}[LIV].{3}[LIV].{1,3}[LIV]',
        r'L.[LIV].{2}[LIV].{1,3}[LIV]',
        r'[LIV].L.{2}[LIV].{1,3}[LIV]',
    ]

    # Relaxed patterns (with F and M)
    patterns_relaxed = [
        r'[LIVFM].{2}[LIVFM].{2}[LIVFM].{1,3}[LIVFM]',
        r'[LIVFM].{3}[LIVFM].{2}[LIVFM].{1,3}[LIVFM]',
        r'[LIVFM].{2}[LIVFM].{3}[LIVFM].{1,3}[LIVFM]',
    ]

    strict_match = any(re.search(p, sequence) for p in patterns_strict)
    relaxed_match = any(re.search(p, sequence) for p in patterns_relaxed)

    if strict_match:
        return True, 'strong', 'Contains L-rich consensus pattern'
    elif relaxed_match:
        return True, 'weak', 'Contains hydrophobic pattern but lacks L preference'
    else:
        return False, 'none', 'No consensus pattern detected'


# Phi (hydrophobic anchor) residue quality weights, informed by:
#  - Kosugi et al. 2008, Traffic ("Nuclear Export Signal Consensus
#    Sequences Defined Using a Localization-Based Yeast Selection System")
#    -- screened 101 CRM1-dependent NESs from a random peptide library and
#    found hydrophobic residues OTHER than leucine were "typically allowed
#    at all positions" (Phi = L, V, I, F, or M), and that this expanded
#    vocabulary covered 99% of their selected NESs. This directly
#    contradicts an L-only anchor definition -- the previous version of
#    this function returned 0.0 for any real NES anchored on Ile/Val/Met/
#    Phe with fewer than 3 actual leucines, even if it matched a textbook
#    spacing pattern otherwise.
#  - Guttler et al. 2010, Nat Struct Mol Biol ("NES consensus redefined by
#    structures of PKI-type and Rev-type nuclear export signals bound to
#    CRM1") -- CRM1's hydrophobic pockets have real, measured size
#    preferences, not "any Phi residue is equally good": Phi1/Phi2 favor
#    LARGE side chains (Leu, Met, Phe, Trp) and bind smaller ones (Ala,
#    Val, Ile) poorly there; Phi3/Phi4 favor Ile/Leu/Met and disfavor large
#    aromatics (Phe, Trp) in that direction.
# This function has no real positional register (unlike the register-
# anchored PSSM feature in nes_ml_predictor_improved.py -- see that
# module's own docstring), so it can't say "this residue occupies Phi2"
# with real confidence. Rather than fake that precision, each hydrophobic
# hit gets a literature-informed QUALITY weight reflecting the general
# "large hydrophobic > small hydrophobic" pattern Guttler found at most
# pocket positions, instead of either an L-only count or treating every
# Phi-vocabulary residue as interchangeable.
PHI_RESIDUE_WEIGHTS = {
    'L': 1.00,  # most common anchor in natural NESs (Kosugi 2008); favored at every CRM1 pocket Guttler tested
    'M': 0.95,  # large; favored at Phi1/Phi2, tolerated at Phi3/Phi4
    'F': 0.90,  # large/aromatic; favored at Phi1/Phi2, specifically disfavored at Phi3/Phi4 (Guttler) --
                # scored as a real but position-restricted anchor, not a universal one
    'I': 0.80,  # favored at Phi3/Phi4, binds poorly at Phi1/Phi2 (Guttler)
    'W': 0.75,  # large aromatic, rare in natural NESs but structurally Phe-like
    'V': 0.55,  # smallest of the accepted Phi set; disfavored at Phi1/Phi2, only weakly tolerated elsewhere
}


def calculate_leucine_anchor_score(sequence):
    """How well this sequence's hydrophobic anchor (Phi) residues match the
    CRM1-binding NES consensus -- position spacing AND residue quality, not
    leucine count alone (rewrite; see PHI_RESIDUE_WEIGHTS above
    for the literature this is based on). Name kept as-is for backward
    compatibility with existing callers/weights (NES_HEURISTIC_WEIGHTS
    references this as 'anchor_score')."""
    if len(sequence) < 8:
        return 0.0

    phi_hits = [(i, aa) for i, aa in enumerate(sequence) if aa in PHI_RESIDUE_WEIGHTS]

    if len(phi_hits) < 3:
        return 0.0

    positions = [pos for pos, _ in phi_hits]

    # Spacing regularity -- unchanged logic, now applied across the full
    # Phi vocabulary rather than leucines only (2-4 residue gaps match the
    # X2/X3 spacers in Kosugi's classes; a gap of 5 gets partial credit).
    spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    spacing_score = 0.0
    for spacing in spacings:
        if 2 <= spacing <= 4:
            spacing_score += 1.0
        elif spacing == 5:
            spacing_score += 0.5
    spacing_score /= len(spacings)

    # Position bonus -- unchanged logic (anchor near the N-terminus, and
    # anchors landing near the canonical Phi0/Phi1/Phi2/Phi3/Phi4 offsets),
    # now checked against any Phi-vocabulary residue, not leucine only.
    position_score = 0.0
    if positions[0] <= 2:
        position_score += 0.3
    expected_positions = [0, 3, 6, 9, 11]
    for exp_pos in expected_positions:
        if any(abs(pos - exp_pos) <= 1 for pos in positions):
            position_score += 0.14

    # residue-quality weight -- mean PHI_RESIDUE_WEIGHTS across the
    # anchor residues actually found, so a run of small/disfavored
    # hydrophobics (e.g. all Val) scores lower than one of large/favored
    # ones (Leu/Met/Phe), matching Guttler's real pocket-affinity findings
    # instead of treating every Phi-vocabulary residue as interchangeable.
    quality_score = sum(PHI_RESIDUE_WEIGHTS[aa] for _, aa in phi_hits) / len(phi_hits)

    # Reweighted to make room for quality_score (previously spacing*0.6 +
    # position*0.4). This 3-way split is a judgment call, not itself
    # literature-derived -- anchor_score carries a small overall weight in
    # NES_HEURISTIC_WEIGHTS (~0.015-0.027) regardless, so it's worth
    # revisiting with diagnose_feature_importance.py after retraining
    # rather than treating this split as final.
    total_score = spacing_score * 0.45 + position_score * 0.30 + quality_score * 0.25
    return min(1.0, total_score)


# =============================================================================
# DATA-DRIVEN NES HEURISTIC SCORING WEIGHTS
# =============================================================================
# The weights below used to be hardcoded literals with comments justifying
# them by hand. That's fragile in two ways: (1) the justification can go
# stale -- it did, when the disorder_score weight was bumped from 0.02 to
# 0.14 based on a diagnostic run that later turned out to be corrupted by a
# data-construction bug (negatives got placeholder flanking features), and
# nobody was forced to revisit the number once the bug was fixed; (2) it
# requires a human to notice the model changed and manually redo this
# analysis every time nes_ml_predictor_improved.py is retrained.
#
# Instead, most of these weights are now DERIVED at startup from
# models/feature_diagnosis/diagnosis_report.json -- the same file
# diagnose_feature_importance.py already writes after every training run.
# Whoever retrains the model and re-runs the diagnostic script just needs to
# restart app.py; the weights below shift automatically, no code edit
# required. If that file is missing, unreadable, or doesn't cover a needed
# feature, everything falls back to the static values worked out by hand
# from the diagnostics available when this was written (see
# NES_STATIC_FALLBACK_WEIGHTS / NES_COMPREHENSIVE_STATIC_FALLBACK_WEIGHTS).
#
# Three terms are deliberately kept FIXED, not derived:
#   - ml_score / ml_norm: this isn't "how important is one feature", it's a
#     standing policy choice about how much to lean on the trained model's
#     overall verdict vs. the hand-engineered terms. The diagnosis report
#     has no single "overall model quality" number to derive this from.
#   - effective_pocket_score / (no comprehensive equivalent): fpocket
#     geometry is never one of the ML model's own input features, so
#     nothing in diagnosis_report.json measures it at all.
#   - confidence_score: deliberately a trust/reliability modifier on the
#     *other* terms, not a discriminative feature in its own right, so it
#     isn't a candidate for "how much does this predict the label" scoring.
#   - flanking_access_norm (comprehensive formula only): flank SASA isn't a
#     trained model feature either (only flank *disorder* is), so there's
#     nothing to derive it from.
#
# IMPORTANT: pattern_score/anchor_score/hydro_score are deliberately mapped
# to the *plain composition* diagnostic features (frac_L, frac_phi_total,
# mean_hydro, phi_pos_*_hydrophobicity,...) and NOT to pssm_score, even
# though pssm_score is the model's single most important feature by far.
# pssm_score's contribution is already fully represented via ml_score's
# fixed weight (ml_score IS the model's output probability, which already
# bakes pssm_score in). Reusing pssm_score's importance again to size
# pattern_score/anchor_score would recreate the exact redundant
# quadruple-counting problem (pattern/anchor/ml/hydro all measuring the same
# leucine-register signal) this reweighting was meant to fix.
NES_DIAGNOSIS_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'feature_diagnosis', 'diagnosis_report.json')

# REVISED AGAIN: ml_score is the trained model's actual output probability,
# computed from ALL 37 of its input features at once -- not "some of what
# it trained on". That set already includes sasa_norm and nes_disorder_mean
# directly (they're literal model inputs), and frac_L/mean_hydro/phi_pos_*
# are exactly what pssm_score is built from. So pattern_score, anchor_score,
# hydro_score, disorder_score, and the additive part of sasa_score were all
# re-adding evidence the model had already weighed in, just to varying
# degrees -- not only the pattern/anchor/hydro trio caught the first time.
# ml_score's fixed weight is raised accordingly (0.40 -> 0.78) and the 5
# dynamically-derived terms now share a much smaller combined budget.
#
# Two things are NOT folded into ml_score, on purpose:
#   - effective_pocket_score: fpocket geometry was never a model input at
#     all, so this is genuinely independent evidence, not a duplicate.
#   - The exposure_factor GATE further down (separate from sasa_score's
#     now-tiny additive weight): the model only learned a soft statistical
#     association between accessibility and being a real NES from ~1200
#     examples. It has no mechanism to enforce a hard physical rule ("fully
#     buried = cannot bind CRM1, full stop"), only correlations -- so the
#     multiplicative gate stays exactly as strong as before. If anything it
#     matters MORE now, since it's the only remaining place enforcing that
#     constraint once the additive sasa_score vote shrinks.
NES_FIXED_WEIGHTS = {
    'ml_score': 0.78,
    'effective_pocket_score': 0.10,
    'confidence_score': 0.02,
}
NES_COMPREHENSIVE_FIXED_WEIGHTS = {
    'ml_norm': 0.75,
    # flanking_access_norm stays meaningful (not folded in) -- flank *SASA*
    # was never a model feature either (only flank *disorder* was), so this
    # is genuinely independent information the model can't already see.
    'flanking_access_norm': 0.15,
}

NES_DYNAMIC_FEATURE_MAP = {
    'pattern_score': ['frac_L', 'frac_phi_total'],
    'anchor_score': ['frac_L'],
    'disorder_score': ['nes_disorder_mean', 'n_flank_disorder', 'c_flank_disorder'],
    'sasa_score': ['sasa_norm'],
    'hydro_score': ['mean_hydro', 'max_hydro', 'phi_pos_0_hydrophobicity',
                     'phi_pos_1_hydrophobicity', 'phi_pos_2_hydrophobicity',
                     'phi_pos_3_hydrophobicity', 'phi_pos_4_hydrophobicity'],
}
NES_COMPREHENSIVE_DYNAMIC_FEATURE_MAP = {
    'sasa_norm': ['sasa_norm'],
    'combined_disorder': ['nes_disorder_mean'],
    'flexibility_norm': ['plddt_norm'],
}

# Static fallback -- must sum to exactly (1 - sum(fixed_weights)) for each
# formula so weights still total 1.0 even when the diagnosis report is
# missing (see the renormalization safety net in _derive_dynamic_weights
# below, which now also protects against this being wrong). Proportions
# mirror the same reasoning as the dynamic path: sasa_score/disorder_score
# get relatively more of the small leftover budget than pattern_score/
# anchor_score/hydro_score, matching what the diagnostics show once
# pssm_score's contribution is excluded from consideration (see
# NES_DYNAMIC_FEATURE_MAP above).
NES_STATIC_FALLBACK_WEIGHTS = {
    'sasa_score': 0.030, 'pattern_score': 0.025, 'disorder_score': 0.020,
    'anchor_score': 0.015, 'hydro_score': 0.010,
}
NES_COMPREHENSIVE_STATIC_FALLBACK_WEIGHTS = {
    'flexibility_norm': 0.045, 'sasa_norm': 0.035, 'combined_disorder': 0.020,
}


def _composite_feature_scores(diagnosis):
    """Blends each feature's 4 independent importance signals (impurity,
    permutation, univariate AUC, point-biserial |r|) into one composite
    share of the model's total decision-relevant signal. Each metric is
    first turned into a fraction of its own total *across all features*
    (so a feature that's #1 by impurity but #10 by permutation gets a
    blended ranking, not a single-method one), then the 4 fractions are
    averaged. Negative permutation importance (shuffling the feature made
    the model *better* -- noise) is clipped to 0 before normalizing; AUC is
    re-centered so only the informative side (>0.5) counts."""
    impurity = diagnosis.get('impurity_importance', {})
    perm = diagnosis.get('permutation_importance_mean', {})
    auc = diagnosis.get('univariate_auc', {})
    biserial = diagnosis.get('point_biserial_correlation', {})
    features = set(impurity) | set(perm) | set(auc) | set(biserial)

    def _fractions(d, transform=lambda v: v):
        vals = {f: max(0.0, transform(d.get(f, 0.0))) for f in features}
        total = sum(vals.values())
        if total <= 0:
            return {f: 0.0 for f in features}
        return {f: v / total for f, v in vals.items()}

    impurity_frac = _fractions(impurity)
    perm_frac = _fractions(perm)
    auc_frac = _fractions(auc, transform=lambda v: (v - 0.5) * 2)
    biserial_frac = _fractions(biserial, transform=lambda v: abs(v))

    return {f: (impurity_frac[f] + perm_frac[f] + auc_frac[f] + biserial_frac[f]) / 4.0
            for f in features}


def _derive_dynamic_weights(fixed_weights, feature_map, static_fallback,
                             diagnosis_path=NES_DIAGNOSIS_REPORT_PATH):
    """Splits (1 - sum(fixed_weights)) across feature_map's terms,
    proportional to each term's mapped diagnostic feature(s)' composite
    importance. Falls back to static_fallback (unchanged) if the diagnosis
    report is missing/unreadable/empty of signal."""
    remaining_budget = 1.0 - sum(fixed_weights.values())
    weights = dict(fixed_weights)

    def _use_fallback(reason):
        print(f"  [scoring weights] {reason} -- using static fallback weights for {list(feature_map)}.")
        # Renormalized defensively so weights always sum to exactly 1.0 even
        # if static_fallback's values don't add up to remaining_budget
        # precisely (e.g. after a future hand-edit that forgets to rebalance
        # them) -- a silent <1.0 total would quietly deflate every score.
        fb_total = sum(static_fallback.values())
        if fb_total > 0:
            weights.update({term: static_fallback[term] * remaining_budget / fb_total for term in feature_map})
        else:
            weights.update({term: remaining_budget / len(feature_map) for term in feature_map})
        return weights

    try:
        with open(diagnosis_path) as f:
            diagnosis = json.load(f)
        composite = _composite_feature_scores(diagnosis)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        return _use_fallback(f"Could not read {diagnosis_path} ({e})")

    term_scores = {}
    for term, mapped_features in feature_map.items():
        scores = [composite.get(f, 0.0) for f in mapped_features]
        term_scores[term] = sum(scores) / len(scores) if scores else 0.0

    total_score = sum(term_scores.values())
    if total_score <= 0:
        return _use_fallback(f"Diagnosis report had no usable signal for {list(feature_map)}")

    # Floor each dynamic term at 20% of its "fair share" so a term never
    # fully vanishes just because this run's diagnostics happened to show
    # near-zero importance for it (a real biological signal can be weak in
    # one dataset snapshot without being permanently irrelevant) -- then
    # renormalize so the budget still sums exactly right after flooring.
    fair_share = remaining_budget / len(feature_map)
    floor = fair_share * 0.2
    raw = {term: remaining_budget * (term_scores[term] / total_score) for term in feature_map}
    raw = {term: max(floor, v) for term, v in raw.items()}
    raw_total = sum(raw.values())
    weights.update({term: v * remaining_budget / raw_total for term, v in raw.items()})
    return weights


NES_HEURISTIC_WEIGHTS = _derive_dynamic_weights(
    NES_FIXED_WEIGHTS, NES_DYNAMIC_FEATURE_MAP, NES_STATIC_FALLBACK_WEIGHTS)
NES_COMPREHENSIVE_WEIGHTS = _derive_dynamic_weights(
    NES_COMPREHENSIVE_FIXED_WEIGHTS, NES_COMPREHENSIVE_DYNAMIC_FEATURE_MAP,
    NES_COMPREHENSIVE_STATIC_FALLBACK_WEIGHTS)

print(f"[scoring weights] NES heuristic weights (data-driven where possible): "
      f"{ {k: round(v, 3) for k, v in NES_HEURISTIC_WEIGHTS.items()} }")
print(f"[scoring weights] NES comprehensive weights (data-driven where possible): "
      f"{ {k: round(v, 3) for k, v in NES_COMPREHENSIVE_WEIGHTS.items()} }")


def calculate_improved_nes_score(sequence, ml_score, hydro_score, disorder_score,
                                 sasa_score, confidence_score, pocket_score,
                                 max_helix_run=None, has_second_helix=None,
                                 uniprot_coiled_coil_overlap=False,
                                 uniprot_domain_overlap=False):
    """
    IMPROVED NES SCORING emphasizing leucine/consensus patterns, real
    disorder, and real exposure -- with fpocket treated as a *secondary,
    exposure-gated* corroborating signal rather than a dominant one.

    Why fpocket can't be trusted on its own: fpocket (and our geometry-based
    fallback) only ever sees a single, static AlphaFold conformation. It has
    no concept of conformational flexibility or rigidity -- it just measures
    concave, hydrophobic-lined cavities in whatever frozen shape the model
    happens to show. A real NES only becomes CRM1-accessible when its region
    is actually flexible/disordered enough to present that hydrophobic face
    at the surface; a rigid, buried hydrophobic pocket deep in a folded
    domain can look geometrically identical to fpocket even though CRM1 can
    never reach it. So pocket_score is discounted by real measured exposure
    (sasa_score) before it's allowed to influence anything below -- a
    "pocket" in a low-SASA region counts for very little, no matter how
    good it looks geometrically.
    """

    # HARD FILTER: Leucine requirement
    leucine_valid, leucine_msg = validate_nes_leucine_requirement(sequence)
    if not leucine_valid:
        return 0.0, {
            'leucine_filter': leucine_msg,
            'status': 'REJECTED - insufficient leucines'
        }

    # Consensus pattern matching
    has_pattern, pattern_strength, pattern_msg = check_consensus_patterns(sequence)

    if pattern_strength == 'strong':
        pattern_score = 1.0
    elif pattern_strength == 'weak':
        pattern_score = 0.6
    else:
        pattern_score = 0.2

    # Leucine anchor positioning
    anchor_score = calculate_leucine_anchor_score(sequence)

    # fpocket/geometry-fallback pockets are static-structure artifacts --
    # only let a detected pocket count to the degree this region is also
    # really measured as solvent-exposed. A "pocket" with sasa_score=0.05
    # contributes almost nothing; one with sasa_score=0.6+ keeps its full
    # value.
    pocket_plausibility = min(1.0, sasa_score * 1.6)
    effective_pocket_score = pocket_score * pocket_plausibility

    # DATA-DRIVEN WEIGHTS (see NES_HEURISTIC_WEIGHTS above). ml_score IS the
    # trained model's own output probability, computed from all 37 of its
    # input features at once -- including sasa_norm and nes_disorder_mean
    # directly, and frac_L/mean_hydro/phi_pos_* (what pattern_score/
    # anchor_score/hydro_score are hand-coded approximations of). So this
    # formula isn't really combining 8 independent opinions -- pattern_score,
    # anchor_score, hydro_score, disorder_score, and sasa_score's additive
    # contribution here are all, to varying degrees, evidence ml_score has
    # already accounted for. ml_score's weight (0.78) reflects that; the
    # other 5 terms share a small residual budget, sized proportionally to
    # whatever independent signal the diagnostics still credit them with.
    # Only effective_pocket_score is genuinely separate (fpocket geometry
    # was never a model input at all).
    combined_score = (
        pattern_score * NES_HEURISTIC_WEIGHTS['pattern_score'] +
        anchor_score * NES_HEURISTIC_WEIGHTS['anchor_score'] +
        ml_score * NES_HEURISTIC_WEIGHTS['ml_score'] +
        disorder_score * NES_HEURISTIC_WEIGHTS['disorder_score'] +
        sasa_score * NES_HEURISTIC_WEIGHTS['sasa_score'] +
        effective_pocket_score * NES_HEURISTIC_WEIGHTS['effective_pocket_score'] +
        hydro_score * NES_HEURISTIC_WEIGHTS['hydro_score'] +
        confidence_score * NES_HEURISTIC_WEIGHTS['confidence_score']
    )

    # REMOVED: a "textbook match" bonus used to live here (+15% when
    # pattern_strength=='strong' and anchor_score>0.7 and ml_score>0.8).
    # Now that pattern_score/anchor_score carry only a small, diagnostics-
    # derived weight in combined_score above (see NES_HEURISTIC_WEIGHTS),
    # gating an extra multiplicative bonus on those same two hand-coded
    # checks agreeing with ml_score was triple-counting the identical
    # leucine-register signal a third time -- once in ml_score (via
    # pssm_score), once in the small additive pattern_score/anchor_score
    # terms, and again here as a bonus for them agreeing with each other.

    # Bonus for a strong, exposure-plausible pocket match. Uses
    # effective_pocket_score (already discounted by real SASA) so a
    # geometrically "strong" pocket buried in a rigid core can no longer
    # trigger this on its own.
    if effective_pocket_score > 0.7:
        combined_score = min(1.0, combined_score * 1.15)
    elif effective_pocket_score > 0.5:
        combined_score = min(1.0, combined_score * 1.07)

    # REAL PHYSICAL GATE: a fully buried motif cannot bind CRM1 no matter how
    # textbook its sequence pattern looks. sasa_score is real RSA, 0-1,
    # residue-specific Tien et al. 2013 normalized (see calculate_sasa()),
    # so ~0 = fully buried, ~1 = fully
    # solvent-exposed). Applied last so it gates the final number rather than
    # being diluted/out-bid by the additive weights above. The mid-band ramp
    # was tightened from the first version of this fix (which reached ~1.0x
    # by sasa_score=0.34, too lenient) -- it now only reaches 1.0x at
    # sasa_score=0.40 and drops off more steeply below that.
    if sasa_score >= 0.40:
        exposure_factor = 1.0
    elif sasa_score >= 0.15:
        exposure_factor = 0.3 + ((sasa_score - 0.15) / 0.25) * 0.7
    else:
        exposure_factor = 0.1 + (sasa_score / 0.15) * 0.2
    combined_score *= exposure_factor

    # REAL STRUCTURAL GATE: a real, CA-coordinate-derived long continuous
    # helix run near the candidate is strong physical evidence of a
    # leucine-zipper/coiled-coil, not a real NES -- CRM1 needs a largely
    # unstructured/exposed peptide (Fung & Chook 2017 found NESs bind in at
    # most "one turn of helix"), while a coiled-coil is a long continuous
    # helix by definition (that's what lets it dimerize).
    #
    # Added after three attempts to teach the ML model this as a
    # soft, LEARNED feature -- a much larger hard-negative training set with
    # real structural backfill, a heptad-periodicity sequence proxy, then
    # this exact real CA-geometry signal as a trained feature -- all failed
    # to move a holdout test's 5 hardest real leucine-zipper negatives at
    # all in the live pipeline, even though max_helix_run's raw signal is
    # strongly discriminating on its own (brute-force AUC 0.845 across the
    # real training data) and earned real permutation importance in the
    # trained model (rank 7 of 46, above pssm_score). The model just doesn't
    # weight it heavily enough to overcome candidates that are ALSO
    # near-maximal on pattern/pssm-type features -- exactly the same "a soft
    # learned association isn't a hard physical rule" gap the sasa
    # exposure_factor gate above exists to close for pocket geometry. This
    # is the same fix, same philosophy: a hard, deterministic downstream
    # discount instead of relying on the classifier to learn it.
    #
    # max_helix_run=None means "not computed for this call" (e.g. caller
    # hasn't wired in real structure yet) -- no penalty, absence of evidence
    # isn't evidence of a coiled-coil. Thresholds are set from this
    # project's own real training-data distribution: real NES positives run
    # mean 13 / median 11 residues of continuous helix near the candidate;
    # real leucine-zipper/coiled-coil hard negatives run mean 32 / median 32.
    #
    # Tried gating this on has_second_helix
    # (_has_packed_second_helix() -- is there a SEPARATE helical segment
    # packed against this one in 3D?) to distinguish a real coiled-coil
    # (Myosin-9's dimeric rod) from a single free-standing structured helix
    # that isn't one (e.g. Neurogenin-3's bHLH helix). REVERTED same day:
    # has_second_helix came back False for nearly every real coiled-coil in
    # the set, because AlphaFold models these as single monomers -- the
    # real dimerization partner simply isn't present in the structure being
    # analyzed at all, so "no second helix found" was never reliable
    # negative evidence. has_second_helix is still computed and included in
    # `details` for visibility, it just doesn't affect the score.
    #
    # Real fix -- UniProt's own 'Coiled coil' /
    # coil-or-zipper-worded Region/Domain annotations (see
    # fetch_uniprot_structural_annotations() docstring), which are computed
    # straight from sequence (or curated) and don't depend on which chain
    # AlphaFold happened to model. Validated against the full holdout set
    # before being wired in here: 8/8 real coiled-coil negatives correctly
    # flagged uniprot_coiled_coil_overlap=True, INCLUDING Myosin-9 (P35579),
    # where max_helix_run still reads 1 and would apply no penalty at all on
    # its own -- and neither of the 2 checkable true positives (CAV-VP1,
    # Neurogenin-3) falsely triggered it. Neurogenin-3 additionally showed
    # uniprot_domain_overlap=True (real 'Domain: bHLH' annotation at its
    # candidate window) with no coiled-coil overlap -- real evidence FOR
    # softening the penalty there, unlike has_second_helix's guess.
    #
    # When neither UniProt signal is available (protein/region not
    # annotated -- true for CAV-VP1, which has no coiled-coil issue at all
    # so this doesn't matter for it in practice) this falls back to the
    # original, unmodified max_helix_run-only ramp -- same safe default as
    # before any of today's changes, not a new guess.
    if uniprot_coiled_coil_overlap:
        # independent, real evidence outranks the (sometimes-broken)
        # geometric measurement -- apply unconditionally
        coiled_coil_factor = 0.25
    elif uniprot_domain_overlap:
        # real evidence this is some OTHER structured domain, not a
        # coiled-coil -- softer discount, still scaled by the measured
        # helix run so a very long helix in a non-coiled-coil domain isn't
        # treated as entirely risk-free
        if max_helix_run is None or max_helix_run <= 15:
            coiled_coil_factor = 1.0
        elif max_helix_run <= 28:
            coiled_coil_factor = 1.0 - ((max_helix_run - 15) / 13.0) * 0.25
        else:
            coiled_coil_factor = 0.70
    elif max_helix_run is None:
        coiled_coil_factor = 1.0
    elif max_helix_run <= 15:
        coiled_coil_factor = 1.0
    elif max_helix_run <= 28:
        coiled_coil_factor = 1.0 - ((max_helix_run - 15) / 13.0) * 0.75
    else:
        coiled_coil_factor = 0.25
    combined_score *= coiled_coil_factor

    details = {
        'leucine_filter': leucine_msg,
        'consensus_pattern': pattern_msg,
        'pattern_strength': pattern_strength,
        'pattern_score': round(pattern_score, 3),
        'anchor_score': round(anchor_score, 3),
        'pocket_score': round(pocket_score, 3),
        'effective_pocket_score': round(effective_pocket_score, 3),  # pocket score after exposure discount
        'exposure_factor': round(exposure_factor, 3),
        'max_helix_run_near_candidate': max_helix_run,
        'has_second_helix_packed_nearby': has_second_helix,
        'uniprot_coiled_coil_overlap': uniprot_coiled_coil_overlap,
        'uniprot_domain_overlap': uniprot_domain_overlap,
        'coiled_coil_factor': round(coiled_coil_factor, 3),
        'status': 'VALID'
    }

    return combined_score, details


def _compute_helical_flags_from_residues(residues):
    """Real, CA-coordinate-derived alpha-helix detection (P-SEA method,
    Labesse et al. 1997, CABIOS 13:291-295) against already-parsed
    Biopython Residue objects -- same geometric criteria as
    nes_data_pipeline/structural_dataset_v2_pipeline.py's
    real_ca_helix_geometry(), reimplemented here rather than imported so
    this route doesn't need a cross-package import for two small pure
    functions, and because this route already has the live Structure
    object in memory (no need to re-parse pdb_content as text).

    Returns a list of bools, same length/order/indexing as `residues`
    (i.e. aligned with `sequence`/plddt_values/sasa_values in this route --
    same 0-based-index convention as start_pos/end_pos below), True where
    that residue's CA(i)-CA(i+3) and CA(i)-CA(i+4) distances both fall in
    the range an ideal continuous alpha helix produces.

    Added: after two sequence-only attempts (bigger hard-negative
    training set; a heptad-periodicity sequence proxy) both failed to move
    the holdout test's hardest leucine-zipper negatives at all, this gives
    the model a genuine 3D-structural signal -- a real leucine zipper is a
    long CONTINUOUS helix (structurally required to dimerize); a real NES
    sits in an otherwise disordered/loop region with at most "one turn of
    helix" contacting CRM1's groove (Fung & Chook 2017).

    a looser version of these tolerances (4.6-6.8 / 5.4-7.2) was
    tried and reverted -- it would have made per-residue "is helical"
    slightly easier to satisfy EVERYWHERE in every protein, not just inside
    real coiled-coils, which risked new false negatives on true positives
    with any ordinary local helical turn. Left at the original, validated
    4.8-6.6 / 5.6-7.0 bounds; see _has_packed_second_helix() and the
    +/-45 flank in the STEP 4 _longest_helix_run() call instead for the
    targeted fix for Myosin-9-style coiled-coils.
    """
    coords = []
    for r in residues:
        try:
            coords.append(r['CA'].coord)
        except KeyError:
            coords.append(None)
    n = len(coords)
    flags = [False] * n
    for i in range(n):
        c0 = coords[i]
        if c0 is None:
            continue
        c3 = coords[i + 3] if i + 3 < n else None
        c4 = coords[i + 4] if i + 4 < n else None
        d13 = float(np.linalg.norm(c0 - c3)) if c3 is not None else None
        d14 = float(np.linalg.norm(c0 - c4)) if c4 is not None else None
        ok13 = d13 is not None and 4.8 <= d13 <= 6.6
        ok14 = d14 is not None and 5.6 <= d14 <= 7.0
        flags[i] = bool(ok13 and ok14)
    return flags


def _find_helical_segments(flags, min_length=7):
    """Contiguous runs of >= min_length consecutive True values in `flags`.
    Returns a list of (start_idx, end_idx) inclusive index pairs -- the
    building block for _has_packed_second_helix() below (need real,
    separate helical stretches, not just isolated helical residues, to
    call something a "second helix")."""
    segments = []
    n = len(flags)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if (j - i + 1) >= min_length:
                segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


def _has_packed_second_helix(residues, helix_flags, start_idx, end_idx,
                              flank=45, min_segment_len=10,
                              min_seq_separation=15, contact_distance=11.0,
                              min_contact_residues=3):
    """Real coiled-coil geometry check, added: does the
    candidate's local long helix run have a SEPARATE helical segment
    elsewhere in the protein whose CA atoms pack against it in 3D space
    (within contact_distance A, and at least min_seq_separation residues
    away in sequence so this isn't just the same helix's own turns)?

    This is what actually distinguishes a coiled-coil -- two or more
    helices wound/packed together, e.g. Myosin-9's dimeric rod domain --
    from a single free-standing long helix that ISN'T a coiled-coil, which
    max_helix_run alone can't tell apart. Requiring the partner segment to
    be at least min_segment_len=10 residues (not just any short helical
    turn) is a deliberate attempt to avoid firing on a bHLH domain's own
    intramolecular helix-loop-helix pair, whose two helices are typically
    shorter than a real coiled-coil's -- but this is an imperfect
    distinction, not a guarantee: a real HLH fold does have two packed
    helices by design, so this check can still fire there. Treat it as
    one more real, imperfect signal to combine with max_helix_run, not as
    a solved case.

    Returns (has_second_helix: bool, n_contact_residues: int).
    """
    n = len(helix_flags)
    lo = max(0, start_idx - flank)
    hi = min(n - 1, end_idx + flank)

    all_segments = _find_helical_segments(helix_flags, min_segment_len)
    local_segments = [s for s in all_segments if s[0] <= hi and s[1] >= lo]
    if not local_segments:
        return False, 0

    # "primary" segment = the longest one overlapping this candidate's flank
    primary = max(local_segments, key=lambda s: s[1] - s[0])

    other_segments = [
        s for s in all_segments
        if s != primary
        and (s[1] < primary[0] - min_seq_separation or s[0] > primary[1] + min_seq_separation)
    ]
    if not other_segments:
        return False, 0

    def ca(i):
        try:
            return residues[i]['CA'].coord
        except (KeyError, IndexError):
            return None

    primary_coords = [c for c in (ca(i) for i in range(primary[0], primary[1] + 1)) if c is not None]
    if not primary_coords:
        return False, 0

    contact_residue_count = 0
    for seg_start, seg_end in other_segments:
        for i in range(seg_start, seg_end + 1):
            c = ca(i)
            if c is None:
                continue
            if any(np.linalg.norm(c - pc) <= contact_distance for pc in primary_coords):
                contact_residue_count += 1

    return contact_residue_count >= min_contact_residues, contact_residue_count


def _longest_helix_run(flags, lo, hi):
    """Longest run of CONSECUTIVE indices in [lo, hi] (clipped to the
    array) that are all True in `flags` -- see
    _compute_helical_flags_from_residues docstring. This, not "helical or
    not" in isolation, is what distinguishes a real coiled-coil's extended
    helix from a real NES's short local helical turn."""
    lo = max(0, lo)
    hi = min(len(flags) - 1, hi)
    best = cur = 0
    for i in range(lo, hi + 1):
        if flags[i]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


@app.route('/api/unified_crm1_nes/<model_id>', methods=['GET'])
def unified_crm1_nes_analysis(model_id):
    """
    UNIFIED CRM1/NES ANALYSIS

    This function combines:
    1. fpocket structural analysis for CRM1 binding pockets
    2. ML-based NES prediction (trained on validated sequences)
    3. Sequence-based pattern matching
    4. Structural features (SASA, disorder, hydrophobicity, pLDDT)

    Returns NES motifs that CAN ACTUALLY BIND CRM1, scored by:
    - ML confidence (is it a valid NES?)
    - Pocket quality (can CRM1 physically bind here?)
    - Structural suitability (disorder, SASA, hydrophobicity)
    """

    print(f"\n{'='*80}")
    print(f"UNIFIED CRM1/NES ANALYSIS: {model_id}")
    print(f"{'='*80}")
    start_time = time.time()

    try:
        # Prefer an explicit ?uniprot_id= query param over
        # parsing model_id (same precedent as get_structure() above).
        # Parsing model_id assumes it's always shaped like
        # "AF-{accession}-F1[-model_vN]", which is NOT true for every
        # AlphaFold DB entry -- confirmed in practice for several
        # coronavirus nucleocapsid proteins, whose real entryId from the
        # AlphaFold API is a bare internal numeric ID (e.g.
        # "AF-0000000365840321") with no accession embedded at all. When
        # that happens, parts[1] silently extracts garbage instead of a
        # real accession, which then breaks structure download, UniProt
        # disorder lookup, and IUPred2A fetching all at once (they all
        # depend on this one value). Passing the accession explicitly
        # sidesteps the parsing entirely for callers that already know it.
        uniprot_id = request.args.get('uniprot_id')
        if not uniprot_id and model_id.startswith('AF-'):
            parts = model_id.split('-')
            if len(parts) >= 2:
                uniprot_id = parts[1]

        # ==========================================
        # STEP 1: Download and Parse Structure
        # ==========================================
        print("\n[1/6] Downloading structure...")

        # ALWAYS ask the AlphaFold API for this accession's
        # CURRENT model version/pdbUrl rather than trusting a '-model_vN'
        # suffix embedded in model_id (or defaulting to v4 when absent).
        # AlphaFold DB periodically re-runs and bumps model versions --
        # confirmed in practice that some entries are already on v6 while
        # this code still assumed v4, which 404s. Falls back to the old
        # string-parsing behavior only if the API lookup fails (e.g. rate
        # limited, or a non-UniProt model_id we can't resolve this way).
        pdb_url = None
        if uniprot_id:
            entries = get_alphafold_structure_info(uniprot_id)
            if entries:
                entry = next((e for e in entries if model_id.startswith(e.get('entryId', '\0'))),
                             entries[0])
                pdb_url = entry.get('pdbUrl')
                print(f"   Current AlphaFold version for {uniprot_id}: v{entry.get('latestVersion')}")
        if not pdb_url:
            print("   Could not resolve a current version via the AlphaFold API -- "
                  "falling back to parsing model_id directly (may be stale).")
            if '-model_v' in model_id:
                base_id = model_id.rsplit('-model_v', 1)[0]
                version = model_id.rsplit('-model_v', 1)[1]
                pdb_url = f"https://alphafold.ebi.ac.uk/files/{base_id}-model_v{version}.pdb"
            else:
                pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"

        print(f"   URL: {pdb_url}")
        response = requests.get(pdb_url, timeout=30)

        if response.status_code != 200:
            error_msg = f'Could not download structure from {pdb_url}: HTTP {response.status_code}'
            print(f"   ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 404

        pdb_content = response.text
        print(f"   Downloaded {len(pdb_content)} bytes")

        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', StringIO(pdb_content))

        # Extract residues and data
        residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':
                        residues.append(residue)

        if len(residues) == 0:
            error_msg = "No standard residues found in structure"
            print(f"   ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 400

        # Get sequence
        sequence = ''.join([three_to_one(r.get_resname()) for r in residues])
        residue_numbers = [r.id[1] for r in residues]

        print(f"   Sequence length: {len(sequence)} residues")

        # Get pLDDT from B-factors
        plddt_values = []
        for residue in residues:
            for atom in residue:
                plddt_values.append(atom.bfactor)
                break  # Just get first atom

        print(f"   Mean pLDDT: {np.mean(plddt_values):.1f}")

        # Real CA-coordinate-derived helix geometry (see
        # _compute_helical_flags_from_residues docstring) -- computed once
        # per protein here, sliced per-candidate below (same pattern as
        # plddt_values/sasa_values).
        helix_flags = _compute_helical_flags_from_residues(residues)
        print(f"   Helical residues (real CA geometry): {sum(helix_flags)}/{len(helix_flags)}")

        # Calculate SASA (consensus RSA -- see calculate_sasa docstring).
        # return_stats=True also gets per-residue consensus_z (z-score vs
        # this chain's own distribution) and agreement_sd (spread across
        # the 3 SASA methods) -- ported from consensus_accessibility.py,
        # previously a standalone script never wired into the live app.
        # Used below to build each prediction's rsa_profile (parallel to
        # cider_profile) for the frontend's RSA dropdown.
        print("   Calculating accessibility (consensus RSA + z-score/SD + raw Å²)...")
        # Also request raw (non-Tien-normalized) consensus SASA,
        # for the new raw_hydrophobic_burial CRM1-affinity feature below --
        # see calculate_sasa()'s docstring for why RSA alone isn't enough
        # for that specific purpose (discards absolute residue-size
        # magnitude, which matters for a buried-surface-area binding proxy).
        sasa_values, sasa_computed, rsa_consensus_z, rsa_agreement_sd, raw_sasa_values = calculate_sasa(
            structure, pdb_text=pdb_content, return_stats=True, return_raw=True)
        if sasa_computed:
            print(f"   Mean RSA: {np.mean(sasa_values):.3f}")
        else:
            print(f"   WARNING: Using neutral RSA fallback - real calculation failed")

        # ==========================================
        # STEP 1.5: Fetch UniProt disorder regions
        # ==========================================
        print("\n[1.5/6] Fetching UniProt disorder annotations...")
        uniprot_disorder_regions = []
        if uniprot_id:
            print(f"   UniProt ID: {uniprot_id}")
            uniprot_disorder_regions = fetch_uniprot_disorder_regions(uniprot_id)
            if uniprot_disorder_regions:
                print(f"   Found {len(uniprot_disorder_regions)} disordered regions")
            else:
                print(f"   No disordered regions annotated in UniProt")
        else:
            print(f"   Could not extract UniProt ID from {model_id}")

        # UniProt structural annotations (coiled-coil/zipper regions, and
        # other structured domains) -- see fetch_uniprot_structural_
        # annotations() docstring. NOT yet used in scoring below; computed
        # here and checked per-candidate in STEP 4 purely for logging, same
        # cautious rollout _has_packed_second_helix originally had.
        uniprot_coiled_coil_regions, uniprot_domain_regions = [], []
        if uniprot_id:
            uniprot_coiled_coil_regions, uniprot_domain_regions = fetch_uniprot_structural_annotations(uniprot_id)

        # IUPred2A/ANCHOR2: real sequence-based disorder + disordered-
        # binding-region prediction. Fetched once here as full per-residue
        # arrays aligned to `sequence`; calculate_enhanced_disorder_score()
        # calls below slice out the relevant window/flank fragment from
        # these, same pattern as plddt_values/local_plddt already used.
        # Falls back to None (-> calculate_enhanced_disorder_score's
        # sequence-propensity fallback) if unavailable.
        iupred_scores_full, anchor2_scores_full = None, None
        if uniprot_id:
            iupred_raw, anchor2_raw, iupred_seq = fetch_iupred2a_scores(uniprot_id)
            iupred_scores_full, anchor2_scores_full = align_iupred_to_structure(
                sequence, iupred_seq, iupred_raw, anchor2_raw)
            if iupred_scores_full:
                print(f"   IUPred2A disorder profile aligned ({len(iupred_scores_full)} residues)"
                      + (" + ANCHOR2" if anchor2_scores_full else ""))
            else:
                print(f"   Warning: IUPred2A unavailable -- calculate_enhanced_disorder_score will use sequence-propensity fallback")

        # ==========================================
        # STEP 2: Run fpocket for CRM1-compatible pockets
        # ==========================================
        print("\n[2/6] Detecting CRM1-compatible binding pockets...")

        pockets = []
        pocket_dict = {}  # Maps residue positions to pocket info

        if pocket_detector is not None:
            try:
                pockets = pocket_detector.detect_pockets(pdb_content)
                print(f"   Found {len(pockets)} CRM1-compatible pockets")

                # Build pocket lookup by residue
                for pocket in pockets:
                    for res_num in pocket.get('residue_numbers', []):
                        if res_num not in pocket_dict:
                            pocket_dict[res_num] = []
                        pocket_dict[res_num].append({
                            'score': pocket.get('crm1_compatibility_score', 0),
                            'pocket_id': pocket.get('id', 'unknown'),
                            'hydrophobic_ratio': pocket.get('hydrophobicity_score', 0),
                            # These three were computed by
                            # _filter_for_crm1_compatibility() but never
                            # threaded past this dict, so the per-candidate
                            # 'pocket_details' in the API response only ever
                            # showed a bare score number -- no way to see
                            # WHY a pocket scored well (real shape match?
                            # charge density? residue composition?) without
                            # reading server logs. Now available on every
                            # candidate's pocket_details.
                            'volume': pocket.get('volume', pocket.get('total_sasa')),
                            'druggability_score': pocket.get('druggability_score', pocket.get('drug_score')),
                            'crm1_compatibility_reasons': pocket.get('crm1_compatibility_reasons', []),
                            # Lets the caller tell whether this
                            # pocket came from real fpocket alpha-sphere
                            # geometry or the weaker geometry-based fallback
                            # (see pocket_detector.py's detect_pockets) --
                            # previously invisible in the API response.
                            'detection_method': pocket.get('detection_method', 'unknown'),
                        })
            except Exception as e:
                print(f"   Warning: fpocket error: {e}")
                print(f"   Continuing with sequence-based analysis only")
        else:
            print("   Warning: Pocket detector not available - using sequence-only mode")

        # Per-residue pocket score array (same shape/indexing as crm1_scores
        # below) so the frontend can offer a "CRM1 Pockets" 3D color mode
        # without needing to reconstruct residue-number overlap logic
        # client-side from the raw pocket list.
        #
        # IMPORTANT: this is discounted by real measured SASA before display,
        # for the same reason the NES scoring formula discounts pocket_score
        # (see calculate_improved_nes_score). fpocket only sees one static,
        # frozen conformation -- it has no idea whether a cavity it finds
        # sits in a rigid, buried core or a genuinely surface-accessible
        # groove. Structured/packed regions are actually *more* likely to
        # contain real geometric cavities (that's just protein folding --
        # helix bundles, sheet cores, etc.), while fully extended/disordered
        # loops (rendered by AlphaFold as loose coil, low pLDDT) have no 3D
        # cavity to detect at all. Left undiscounted, this color mode was
        # showing raw geometric "does a dimple exist here" rather than "can
        # CRM1 actually reach this", which is exactly why the rigid, packed
        # middle of a structure could still light up strongly here even
        # after the NES-likelihood scoring itself was fixed to discount
        # buried pockets -- that earlier fix only touched the final NES
        # prediction score, not this separate raw visualization array.
        raw_pocket_scores = np.zeros(len(residue_numbers))
        for i, res_num in enumerate(residue_numbers):
            if res_num in pocket_dict:
                raw_pocket_scores[i] = max(p['score'] for p in pocket_dict[res_num])

        pocket_scores = np.zeros(len(residue_numbers))
        for i in range(len(residue_numbers)):
            if sasa_values is not None and len(sasa_values) > i:
                sasa_score_i = min(1.0, sasa_values[i])  # already RSA (Tien-normalized), no flat divisor
            else:
                sasa_score_i = 0.5
            exposure_plausibility = min(1.0, sasa_score_i * 1.6)
            pocket_scores[i] = raw_pocket_scores[i] * exposure_plausibility

        # ==========================================
        # STEP 3: Scan for NES motifs with ML
        # ==========================================
        print("\n[3/6] Scanning for NES motifs with ML...")

        # Scan with multiple window sizes (9-15 residues is typical for NES).
        # 8 used to be included here, but every consensus
        # regex in nes_ml_predictor_improved.py's NES_PATTERNS needs a
        # minimum 9-residue span (e.g. class_2 = [LIVFM].[LIVFM].{2}[LIVFM].[LIVFM],
        # 9 chars). Any 8-mer that cleared validate_nes_leucine_requirement
        # below could still be accepted as a "candidate" but could never get
        # a real class label -- it always showed up as Class: unknown in the
        # UI, and was exactly the kind of short, low-hydrophobic-content
        # window that scored unexpectedly high after a retrain.
        window_sizes = [9, 10, 11, 12, 13, 14, 15]
        candidate_nes_motifs = []
        rejected_count = 0  # Track leucine filter rejections

        for window_size in window_sizes:
            for i in range(len(sequence) - window_size + 1):
                subseq = sequence[i:i+window_size]
                start_pos = i
                end_pos = i + window_size - 1

                # === STRICT PRE-FILTER: Check leucine requirement ===
                leucine_valid, _ = validate_nes_leucine_requirement(subseq)
                if not leucine_valid:
                    rejected_count += 1
                    continue  # Skip this candidate entirely

                # Get local structural features
                local_plddt = plddt_values[start_pos:end_pos+1] if len(plddt_values) > end_pos else None
                local_sasa = sasa_values[start_pos:end_pos+1] if len(sasa_values) > end_pos else None
                # Real CA-geometry helix run near this candidate. Left at the
                # original +/-20 flank HERE -- this value feeds
                # ml_predictor.predict()'s trained 'max_helix_run_norm'
                # feature (see nes_ml_predictor_improved.py), whose weight
                # was learned against +/-20 statistics; widening it here would
                # quietly shift that one feature's distribution away from
                # what the model was trained on. The +/-45 widening (for the
                # Myosin-9-style "one local kink zeroes out the whole run"
                # problem) is applied only in STEP 4 below, where
                # coiled_coil_factor is a hand-tuned deterministic multiplier,
                # not a trained weight -- no train/inference mismatch there.
                local_helix_run = _longest_helix_run(helix_flags, start_pos - 20, end_pos + 20)

                # ML prediction with improved predictor
                if ml_predictor is not None:
                    try:
                        # Use improved predictor with flanking analysis
                        ml_prob, ml_conf, ml_details = ml_predictor.predict(
                            subseq,
                            full_sequence=sequence,  # For flanking region analysis
                            nes_start=start_pos,     # Position in protein
                            plddt=local_plddt,
                            sasa=local_sasa,
                            max_helix_run=local_helix_run
                        )
                    except Exception as e:
                        print(f"   WARNING: ML prediction failed for {subseq}: {e}")
                        ml_prob = simple_nes_score(subseq)
                        ml_conf = 'medium' if ml_prob > 0.5 else 'low'
                        ml_details = {}
                else:
                    # Fallback: simple pattern matching
                    ml_prob = simple_nes_score(subseq)
                    ml_conf = 'medium' if ml_prob > 0.5 else 'low'
                    ml_details = {}

                # RAISED threshold from 0.4 to 0.5
                if ml_prob > 0.5:
                    candidate_nes_motifs.append({
                        'sequence': subseq,
                        'start': residue_numbers[start_pos],
                        'end': residue_numbers[end_pos],
                        'start_idx': start_pos,
                        'end_idx': end_pos,
                        'ml_probability': ml_prob,
                        'ml_confidence': ml_conf,
                        'ml_details': ml_details,  # Store detailed ML analysis
                        'length': window_size,
                        'local_plddt': local_plddt,
                        'local_sasa': local_sasa
                    })

        print(f"   Rejected {rejected_count} candidates due to insufficient leucines")
        print(f"   Found {len(candidate_nes_motifs)} candidate NES motifs (ML prob > 0.5)")

        # ==========================================
        # STEP 4: Score NES motifs for CRM1 binding
        # ==========================================
        print("\n[4/6] Scoring NES motifs for CRM1 binding potential...")

        unified_predictions = []

        for motif in candidate_nes_motifs:
            seq = motif['sequence']
            start_idx = motif['start_idx']
            end_idx = motif['end_idx']
            start_pos = motif['start']
            end_pos = motif['end']

            # === SCORING COMPONENTS ===

            # 1. ML Score (is it a valid NES pattern?)
            ml_score = motif['ml_probability']

            # 2. Hydrophobicity (NES needs hydrophobic anchors)
            hydro_scores = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
            avg_hydro = np.mean(hydro_scores) if hydro_scores else 0
            max_hydro = np.max(hydro_scores) if hydro_scores else 0
            hydro_score = (avg_hydro / 4.5) * 0.5 + (max_hydro / 4.5) * 0.5  # Normalize
            hydro_score = max(0, min(1, hydro_score))

            # 3. ENHANCED Disorder propensity -- IUPred2A (+ ANCHOR2) when
            # available, sliced to this candidate's window by array index
            # (start_idx/end_idx), NOT start_pos/end_pos (those are PDB
            # residue numbers, used only for the UniProt-region check inside
            # the function -- see calculate_enhanced_disorder_score).
            iupred_window = (iupred_scores_full[start_idx:end_idx + 1]
                              if iupred_scores_full else None)
            anchor2_window = (anchor2_scores_full[start_idx:end_idx + 1]
                               if anchor2_scores_full else None)
            disorder_score, disorder_details = calculate_enhanced_disorder_score(
                seq, start_pos, end_pos, motif['local_plddt'], uniprot_disorder_regions,
                iupred_fragment=iupred_window, anchor2_fragment=anchor2_window
            )

            # Real N-/C-flank SASA + disorder, for the frontend's "Flanking
            # Region" panel -- this used to always show N/A because the ML
            # predictor's own flanking_analysis (hpr/nc/likelihoods) is
            # purely sequence-based and never had real structural SASA/pLDDT
            # to work with. Computed here instead, using the same +/-15
            # residue window the predictor's own n_flank_disorder/
            # c_flank_disorder features use, for consistency.
            flank_window = 15
            n_flank_start_idx = max(0, start_idx - flank_window)
            c_flank_end_idx = min(len(sequence), end_idx + 1 + flank_window)

            n_flank_sasa_vals = sasa_values[n_flank_start_idx:start_idx] if sasa_values else []
            c_flank_sasa_vals = sasa_values[end_idx + 1:c_flank_end_idx] if sasa_values else []
            flank_sasa_vals = list(n_flank_sasa_vals) + list(c_flank_sasa_vals)
            flank_sasa_avg = float(np.mean(flank_sasa_vals)) if flank_sasa_vals else None

            n_flank_seq = sequence[n_flank_start_idx:start_idx]
            c_flank_seq = sequence[end_idx + 1:c_flank_end_idx]
            n_flank_plddt_vals = plddt_values[n_flank_start_idx:start_idx] if plddt_values else []
            c_flank_plddt_vals = plddt_values[end_idx + 1:c_flank_end_idx] if plddt_values else []

            n_flank_iupred = (iupred_scores_full[n_flank_start_idx:start_idx]
                               if iupred_scores_full else None)
            n_flank_anchor2 = (anchor2_scores_full[n_flank_start_idx:start_idx]
                                if anchor2_scores_full else None)
            c_flank_iupred = (iupred_scores_full[end_idx + 1:c_flank_end_idx]
                               if iupred_scores_full else None)
            c_flank_anchor2 = (anchor2_scores_full[end_idx + 1:c_flank_end_idx]
                                if anchor2_scores_full else None)

            flank_disorder_vals = []
            if n_flank_seq:
                d, _ = calculate_enhanced_disorder_score(
                    n_flank_seq, residue_numbers[n_flank_start_idx], residue_numbers[start_idx - 1],
                    n_flank_plddt_vals, uniprot_disorder_regions,
                    iupred_fragment=n_flank_iupred, anchor2_fragment=n_flank_anchor2
                )
                flank_disorder_vals.append(d)
            if c_flank_seq:
                d, _ = calculate_enhanced_disorder_score(
                    c_flank_seq, residue_numbers[end_idx + 1], residue_numbers[c_flank_end_idx - 1],
                    c_flank_plddt_vals, uniprot_disorder_regions,
                    iupred_fragment=c_flank_iupred, anchor2_fragment=c_flank_anchor2
                )
                flank_disorder_vals.append(d)
            flank_disorder_avg = float(np.mean(flank_disorder_vals)) if flank_disorder_vals else None

            # 4. Surface accessibility (NES must be surface-exposed)
            if motif['local_sasa']:
                avg_sasa = np.mean(motif['local_sasa'])
                sasa_score = min(1.0, avg_sasa)  # already RSA (Tien-normalized), no flat divisor
            else:
                sasa_score = 0.5  # Unknown

            # 5. Confidence (higher pLDDT regions are more reliable)
            if motif['local_plddt']:
                avg_plddt = np.mean(motif['local_plddt'])
                confidence_score = avg_plddt / 100
            else:
                confidence_score = 0.7  # Default

            # 6. NEW: Flexibility score (low pLDDT = high flexibility, good for NES)
            if motif['local_plddt']:
                avg_plddt = np.mean(motif['local_plddt'])
                # pLDDT 40-70 is ideal for NES (structured but flexible)
                # Score peaks around pLDDT 50-60
                if 50 <= avg_plddt <= 70:
                    flexibility_score = 1.0
                elif 40 <= avg_plddt < 50:
                    flexibility_score = 0.8 + (avg_plddt - 40) * 0.02
                elif 70 < avg_plddt <= 80:
                    flexibility_score = 1.0 - (avg_plddt - 70) * 0.05
                elif avg_plddt < 40:
                    flexibility_score = 0.5  # Too disordered
                else:  # > 80
                    flexibility_score = 0.3  # Too rigid
            else:
                flexibility_score = 0.5

            # 7. Pocket compatibility (can CRM1 actually bind here?)
            pocket_score = 0
            pocket_info = []
            crm1_binding_affinity = 0  # CRM1-specific binding score

            for res_idx in range(start_idx, end_idx + 1):
                res_num = residue_numbers[res_idx]
                if res_num in pocket_dict:
                    # This residue is in a CRM1-compatible pocket
                    for pocket in pocket_dict[res_num]:
                        pocket_score = max(pocket_score, pocket['score'])
                        # Calculate CRM1 binding affinity from pocket properties
                        affinity = pocket['score'] * pocket.get('hydrophobic_ratio', 0.5)
                        crm1_binding_affinity = max(crm1_binding_affinity, affinity)
                        pocket_info.append(pocket)

            has_pocket = pocket_score > 0.3

            # Raw (absolute Å²) hydrophobic burial -- a
            # physically-grounded CRM1 binding-affinity signal RSA can't
            # provide on its own (see calculate_sasa()'s docstring: RSA
            # deliberately discards absolute magnitude to remove the
            # residue-size confound, but that magnitude is exactly what
            # correlates with buried-surface-area binding energy, Chothia
            # 1976, ~25 cal/mol per Å²). Sums real raw SASA over just this
            # candidate's Φ-type anchor residues (PHI_RESIDUE_WEIGHTS:
            # L/M/F/I/W/V, the same literature-grounded set used by
            # calculate_leucine_anchor_score) rather than the whole span,
            # since only those residues would actually insert into CRM1's
            # groove. RAW_BURIAL_REFERENCE_A2 (~4 well-exposed bulky anchors,
            # Trp/Phe max ASA ~210-230 Å², Tien et al. 2013) is a heuristic
            # comparability scale, not a precise ΔG calculation.
            RAW_BURIAL_REFERENCE_A2 = 600.0
            anchor_raw_area = 0.0
            if sasa_computed and raw_sasa_values is not None:
                anchor_raw_area = sum(
                    raw_sasa_values[res_idx] for res_idx in range(start_idx, end_idx + 1)
                    if sequence[res_idx] in PHI_RESIDUE_WEIGHTS
                )
            raw_hydrophobic_burial_score = min(1.0, anchor_raw_area / RAW_BURIAL_REFERENCE_A2)

            # NOT blended into crm1_binding_affinity (was 70/30
            # fpocket/burial through. evaluate_crm1_pocket_signal.py
            # tested both signals against real labeled NES positives (up to
            # 128, all AlphaFold-resolvable + UniProt-sequence-verified) and
            # real hard negatives (198 coiled_coil/leucine_zipper decoys),
            # using the real CRM1AwarePocketDetector + real fpocket runs
            # (detection_method confirmed 100% 'fpocket', not the geometry
            # fallback, across the whole clean run). Result, replicated
            # across two independent clean samples (n=36 and n=128
            # positives): fpocket-based affinity correlates in the expected
            # direction (mean 0.41 in positives vs 0.34 in negatives, AUC
            # ~0.56), but raw_hydrophobic_burial is backwards -- HIGHER in
            # hard negatives than real NES (mean 0.56 positives vs 0.70
            # negatives, AUC ~0.57 the wrong way round) in both runs. Neither
            # individually clears p<0.05 at this sample size, but the
            # consistent direction across independent samples is real signal,
            # not noise, and it doesn't support burial as a positive
            # contributor to a "binding affinity" field. Still computed and
            # reported below (raw_hydrophobic_burial component) since it's
            # informative on its own -- just not blended in as if it agreed
            # with the pocket-detection estimate.

            # === USE IMPROVED SCORING ===
            # Real CA-geometry helix run near this candidate. Widened from
            # +/-20 to +/-45 on -- see the matching comment at the
            # other _longest_helix_run() call site above (STEP 3 scan) for
            # why: a narrower flank was measurably zeroed out by one local
            # irregularity on a real coiled-coil (Myosin-9), letting a true
            # negative through undiscounted. Safe to widen only HERE (not in
            # STEP 3) because coiled_coil_factor is a hand-tuned deterministic
            # multiplier, not a trained ML feature -- no train/inference
            # mismatch risk.
            motif_helix_run = _longest_helix_run(helix_flags, start_idx - 45, end_idx + 45)
            # real coiled-coil geometry check -- is there a
            # SEPARATE helical segment elsewhere in the protein actually
            # packed against this one in 3D? See _has_packed_second_helix()
            # docstring. Lets coiled_coil_factor tell a genuine two-helix
            # coiled-coil (Myosin-9's rod) apart from a single long
            # free-standing helix that isn't one (e.g. Neurogenin-3's bHLH
            # helix) -- max_helix_run alone can't make that distinction.
            has_second_helix, n_helix_contacts = _has_packed_second_helix(
                residues, helix_flags, start_idx, end_idx
            )
            # UniProt structural-annotation overlap check --
            # see fetch_uniprot_structural_annotations() docstring. Real
            # residue numbering (start_pos/end_pos), not the 0-based
            # sequence index used for the CA-geometry checks above. NOT
            # wired into coiled_coil_factor yet -- computed and logged only,
            # pending validation against a full holdout run (the same
            # caution _has_packed_second_helix should have gotten before it
            # caused the regression).
            uniprot_coiled_coil_overlap = _overlaps_any_region(
                start_pos, end_pos, uniprot_coiled_coil_regions
            )
            uniprot_domain_overlap = _overlaps_any_region(
                start_pos, end_pos, uniprot_domain_regions
            )
            # Sequence-based heptad-repeat 'peakiness' -- see
            # _calculate_heptad_periodicity's docstring (already validated,
            # brute-force AUC 0.64, real coiled-coils score higher). Logged
            # here as the intended FALLBACK signal for proteins UniProt has
            # no Coiled coil/Domain annotation for at all -- a continuous
            # score to blend in gently rather than another hard gate,
            # deliberately not another binary check after the last one.
            # start_idx/end_idx (0-based), same convention ml_predictor
            # already uses internally -- see STEP 3's ml_predictor.predict()
            # call above.
            heptad_score = None
            if ml_predictor is not None:
                try:
                    heptad_score = ml_predictor._calculate_heptad_periodicity(
                        sequence, start_idx, end_idx
                    )
                except Exception:
                    heptad_score = None
            combined_score, score_details = calculate_improved_nes_score(
                seq, ml_score, hydro_score, disorder_score,
                sasa_score, confidence_score, pocket_score,
                max_helix_run=motif_helix_run,
                has_second_helix=has_second_helix,
                uniprot_coiled_coil_overlap=uniprot_coiled_coil_overlap,
                uniprot_domain_overlap=uniprot_domain_overlap,
            )
            score_details['heptad_periodicity'] = round(heptad_score, 3) if heptad_score is not None else None

            # Skip if rejected by scoring
            if score_details['status'] == 'REJECTED - insufficient leucines':
                continue

            # NOTE: this used to apply two MORE bonus multipliers here
            # (+15% for has_pocket, +10% for flexibility_score>0.8) on top of
            # what calculate_improved_nes_score() already returns. That was
            # a real bug: those bonuses were applied to combined_score
            # *after* the function's own exposure_factor gate had already
            # run, so they weren't discounted for buried/rigid regions at
            # all -- a deeply-buried candidate that got correctly suppressed
            # by the exposure gate could then get re-inflated by up to
            # ~1.27x completely ungated, which is a big part of why
            # predictions kept showing up in inaccessible regions, and why
            # pocket detection ended up dominating everything else once real
            # pocket residue mapping started working. pocket_score and
            # flexibility are now accounted for (and properly
            # exposure-discounted) inside calculate_improved_nes_score
            # itself, so no further un-gated boost is applied here.

            # ANCHOR2 bonus (same fixed, hand-picked weight as the
            # /api/crm1_analysis comprehensive-scoring path, for
            # consistency between the two NES endpoints) -- ANCHOR2 was
            # never a trained-model feature, so it isn't part of
            # calculate_improved_nes_score()'s data-derived NES_HEURISTIC_WEIGHTS.
            if anchor2_window:
                combined_score = min(1.0, combined_score + float(np.mean(anchor2_window)) * 0.08)

            # CIDER linear profiles (hydropathy / NCPR / FCR / complexity)
            # computed over this NES candidate plus a +/-20 residue flanking
            # window of REAL protein context, so the graphs attached to each
            # prediction show the actual charge/hydropathy pattern around
            # the motif, not just the bare 8-15 residue window itself.
            cider_flank = 20
            cider_ctx_start = max(0, start_idx - cider_flank)
            cider_ctx_end = min(len(sequence), end_idx + 1 + cider_flank)
            cider_ctx_seq = sequence[cider_ctx_start:cider_ctx_end]
            cider_raw = compute_linear_cider_profiles(cider_ctx_seq)
            cider_profile = {
                'positions': [residue_numbers[cider_ctx_start + k] for k in range(len(cider_ctx_seq))],
                'linear_hydropathy': cider_raw['linear_hydropathy'],
                'linear_ncpr': cider_raw['linear_ncpr'],
                'linear_fcr': cider_raw['linear_fcr'],
                'linear_complexity': cider_raw['linear_complexity'],
                'cider_computed': cider_raw['cider_computed'],
                # index (within the arrays above) where the predicted NES
                # itself starts/ends, so the frontend can shade that span
                'nes_start_idx_in_profile': start_idx - cider_ctx_start,
                'nes_end_idx_in_profile': end_idx - cider_ctx_start,
            }

            # RSA consensus/z-score/SD profile (ported from
            # consensus_accessibility.py -- see calculate_sasa(return_stats=True)
            # docstring) over this NES candidate plus the same +/-20 residue
            # flanking window used for cider_profile, so the frontend can plot
            # real per-residue solvent accessibility (not just the single
            # averaged sasa_score already in 'components') alongside how
            # unusual each residue's exposure is relative to the rest of this
            # protein (consensus_z) and how much the 3 SASA methods agree
            # (agreement_sd, low = trustworthy).
            rsa_flank = 20
            rsa_ctx_start = max(0, start_idx - rsa_flank)
            rsa_ctx_end = min(len(sequence), end_idx + 1 + rsa_flank)
            rsa_profile = {
                'positions': [residue_numbers[k] for k in range(rsa_ctx_start, rsa_ctx_end)],
                'consensus_rsa': [round(v, 3) for v in sasa_values[rsa_ctx_start:rsa_ctx_end]],
                'consensus_z': [round(v, 3) if v is not None else None
                                for v in rsa_consensus_z[rsa_ctx_start:rsa_ctx_end]],
                'agreement_sd': [round(v, 3) if v is not None else None
                                 for v in rsa_agreement_sd[rsa_ctx_start:rsa_ctx_end]],
                'rsa_computed': sasa_computed,
                'nes_start_idx_in_profile': start_idx - rsa_ctx_start,
                'nes_end_idx_in_profile': end_idx - rsa_ctx_start,
            }

            # Real per-residue IUPred2A (disorder) + ANCHOR2
            # (disordered-binding-region) profile, same +/-20 window as
            # cider_profile/rsa_profile above. iupred_scores_full/
            # anchor2_scores_full (already fetched + structure-aligned once
            # per protein, see align_iupred_to_structure) were previously
            # only ever collapsed into scalar window-mean summaries
            # (disorder_details' sequence_disorder/anchor2_binding) -- never
            # exposed as a real per-residue profile the way CIDER/RSA are,
            # so there was nothing for a per-residue IUPred/ANCHOR2 plot to
            # read. iupred_computed=False (both arrays None) happens
            # whenever iupred2a.elte.hu was unreachable or this protein's
            # structure sequence couldn't be aligned to a fetched IUPred2A
            # profile -- see align_iupred_to_structure's docstring.
            iupred_ctx_start = cider_ctx_start  # identical +/-20 window, reuse rather than recompute
            iupred_ctx_end = cider_ctx_end
            iupred_profile = {
                'positions': [residue_numbers[k] for k in range(iupred_ctx_start, iupred_ctx_end)],
                'iupred2': ([round(v, 3) for v in iupred_scores_full[iupred_ctx_start:iupred_ctx_end]]
                            if iupred_scores_full else None),
                'anchor2': ([round(v, 3) for v in anchor2_scores_full[iupred_ctx_start:iupred_ctx_end]]
                            if anchor2_scores_full else None),
                'iupred_computed': iupred_scores_full is not None,
                'anchor2_computed': anchor2_scores_full is not None,
                'nes_start_idx_in_profile': start_idx - iupred_ctx_start,
                'nes_end_idx_in_profile': end_idx - iupred_ctx_start,
            }

            unified_predictions.append({
                'sequence': seq,
                'start': motif['start'],
                'end': motif['end'],
                'length': motif['length'],
                'combined_score': round(combined_score, 3),
                'components': {
                    'ml_probability': round(ml_score, 3),
                    'ml_confidence': motif['ml_confidence'],
                    # Enhanced ML details from improved predictor
                    'nes_classes': motif.get('ml_details', {}).get('nes_classes', []),
                    'pssm_score': motif.get('ml_details', {}).get('pssm_score', 0),
                    'spacer_hydrophobicity': motif.get('ml_details', {}).get('spacer_hydrophobicity', 0),
                    'flanking_analysis': {
                        **(motif.get('ml_details', {}).get('flanking_analysis') or {}),
                        # real structural values -- the ML predictor's own
                        # flanking_analysis (hpr/nc/likelihoods) is purely
                        # sequence-based and never had these; the frontend
                        # panel always showed N/A for them until now.
                        'sasa': round(flank_sasa_avg, 2) if flank_sasa_avg is not None else None,
                        'disorder': round(flank_disorder_avg, 3) if flank_disorder_avg is not None else None,
                    } if motif.get('ml_details', {}).get('flanking_analysis') or flank_sasa_avg is not None or flank_disorder_avg is not None else None,
                    # Original components
                    'hydrophobicity': round(hydro_score, 3),
                    'disorder': round(disorder_score, 3),
                    'disorder_details': disorder_details,  # breakdown
                    'anchor2_binding': disorder_details.get('anchor2_binding'),  # convenience top-level, same value as disorder_details
                    'surface_accessibility': round(sasa_score, 3),
                    'structural_confidence': round(confidence_score, 3),
                    'flexibility': round(flexibility_score, 3),  # NEW
                    'pocket_compatibility': round(pocket_score, 3),
                    'crm1_binding_affinity': round(crm1_binding_affinity, 3),  # NEW
                    # Absolute (Å²) hydrophobic burial for this
                    # candidate's anchor residues -- see the comment where
                    # this is computed above. NOT part of crm1_binding_affinity
                    # Found empirically backwards-correlated
                    # with real NES status, see comment above) -- reported
                    # standalone for transparency, not as an agreeing signal.
                    'raw_hydrophobic_burial': {
                        'anchor_raw_area_A2': round(anchor_raw_area, 1),
                        'reference_A2': RAW_BURIAL_REFERENCE_A2,
                        'score': round(raw_hydrophobic_burial_score, 3),
                    },
                    **score_details  # Include leucine and pattern details
                },
                'has_crm1_pocket': has_pocket,
                'pocket_details': pocket_info if has_pocket else None,
                'cider_profile': cider_profile,
                'rsa_profile': rsa_profile,
                'iupred_profile': iupred_profile,
            })

        # Sort by combined score
        unified_predictions.sort(key=lambda x: x['combined_score'], reverse=True)

        # ==========================================
        # STEP 5: Filter and rank predictions
        # ==========================================
        print("\n[5/6] Filtering and ranking predictions...")

        # Remove overlaps - keep highest scoring
        filtered_predictions = []
        used_positions = set()

        for pred in unified_predictions:
            # Check if this region overlaps with already selected ones
            overlap = False
            for pos in range(pred['start'], pred['end'] + 1):
                if pos in used_positions:
                    overlap = True
                    break

            # LOWERED threshold from 0.55 to 0.45 to catch more valid NES
            if not overlap and pred['combined_score'] > 0.45:
                filtered_predictions.append(pred)
                for pos in range(pred['start'], pred['end'] + 1):
                    used_positions.add(pos)

        # Classify by confidence - ADJUSTED thresholds (lowered from 0.75/0.65/0.55)
        high_confidence = [p for p in filtered_predictions if p['combined_score'] > 0.70]
        medium_confidence = [p for p in filtered_predictions if 0.55 < p['combined_score'] <= 0.70]
        low_confidence = [p for p in filtered_predictions if 0.45 < p['combined_score'] <= 0.55]

        print(f"\n   High confidence (score > 0.70):   {len(high_confidence)}")
        print(f"   Medium confidence (0.55-0.70):    {len(medium_confidence)}")
        print(f"   Low confidence (0.45-0.55):       {len(low_confidence)}")

        # ==========================================
        # STEP 6: Generate summary statistics
        # ==========================================
        print("\n[6/6] Generating summary...")

        # Calculate average scores
        if filtered_predictions:
            avg_crm1_affinity = np.mean([p['components']['crm1_binding_affinity']
                                         for p in filtered_predictions])
            avg_flexibility = np.mean([p['components']['flexibility']
                                       for p in filtered_predictions])
            avg_disorder = np.mean([p['components']['disorder']
                                   for p in filtered_predictions])
            num_in_uniprot_disorder = sum(1 for p in filtered_predictions
                                          if p['components']['disorder_details']['in_uniprot_disorder_region'])
        else:
            avg_crm1_affinity = 0
            avg_flexibility = 0
            avg_disorder = 0
            num_in_uniprot_disorder = 0

        # Create per-residue scores for visualization
        crm1_scores = np.zeros(len(sequence))
        for pred in filtered_predictions:
            start_idx = pred['start'] - residue_numbers[0]
            end_idx = pred['end'] - residue_numbers[0] + 1
            for i in range(start_idx, min(end_idx, len(crm1_scores))):
                crm1_scores[i] = max(crm1_scores[i], pred['combined_score'])

        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Analysis complete in {elapsed:.1f}s")
        print(f"Average CRM1 binding affinity: {avg_crm1_affinity:.3f}")
        print(f"Average flexibility score: {avg_flexibility:.3f}")
        print(f"Average disorder score: {avg_disorder:.3f}")
        print(f"NES in UniProt disorder regions: {num_in_uniprot_disorder}/{len(filtered_predictions)}")
        print(f"{'='*80}\n")

        return jsonify({
            'crm1_scores': crm1_scores.tolist(),
            'pocket_scores': pocket_scores.tolist(),
            'pockets': [
                {
                    'pocket_id': p.get('id', 'unknown'),
                    'residue_numbers': p.get('residue_numbers', []),
                    'crm1_compatibility': p.get('crm1_compatibility_score', 0),
                    'hydrophobic_ratio': p.get('hydrophobicity_score', 0),
                }
                for p in pockets
            ],
            'nes_motifs': filtered_predictions,
            'crm1_binding_regions': filtered_predictions,  # They're the same now!
            'summary': {
                'total_candidates': len(candidate_nes_motifs),
                'filtered_predictions': len(filtered_predictions),
                'high_confidence': len(high_confidence),
                'medium_confidence': len(medium_confidence),
                'low_confidence': len(low_confidence),
                'pockets_detected': len(pockets),
                # 'fpocket' vs 'geometry_fallback' -- previously
                # invisible in the API response; the only sign anything had
                # fallen back was a startup-time print buried in server
                # logs. 'mixed' can legitimately happen (fpocket succeeds
                # for some structures/runs and not others is not possible
                # within one call, but kept as a safety label in case any
                # pocket is missing the key from an older code path).
                'pocket_detection_method': (
                    pockets[0].get('detection_method', 'unknown') if pockets
                    else ('fpocket' if (pocket_detector is not None and pocket_detector.fpocket_path) else 'no_fpocket_binary')
                ),
                'rejected_by_leucine_filter': rejected_count,
                'analysis_time': round(elapsed, 2),
                'avg_crm1_binding_affinity': round(avg_crm1_affinity, 3),
                'avg_flexibility': round(avg_flexibility, 3),
                'avg_disorder': round(avg_disorder, 3),
                'nes_in_uniprot_disorder': num_in_uniprot_disorder,
                'uniprot_disorder_regions': len(uniprot_disorder_regions),
                'sasa_computed': sasa_computed  # False = fallback placeholder used, not real exposure data
            }
        })

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR in unified CRM1/NES analysis: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


def simple_nes_score(sequence):
    """Simple fallback NES scoring when ML is not available"""
    # Check leucine requirement first
    leucine_valid, _ = validate_nes_leucine_requirement(sequence)
    if not leucine_valid:
        return 0.1  # Very low score

    # Check consensus patterns
    has_pattern, pattern_strength, _ = check_consensus_patterns(sequence)

    if pattern_strength == 'strong':
        base_score = 0.7
    elif pattern_strength == 'weak':
        base_score = 0.5
    else:
        return 0.2

    # Count hydrophobic residues
    hydrophobic_count = sum(sequence.count(aa) for aa in 'LIVFM')
    hydrophobic_ratio = hydrophobic_count / len(sequence)

    # Penalize for too many charged residues
    charged_count = sum(sequence.count(aa) for aa in 'KRDE')
    charged_penalty = charged_count / len(sequence)

    score = base_score + (hydrophobic_ratio * 0.2) - (charged_penalty * 0.2)
    return max(0, min(1, score))


# Tien et al. 2013, theoretical MaxASA (Ų) -- PLoS ONE 8(11):e80635.
# Used to convert raw per-residue SASA into RELATIVE solvent accessibility
# (RSA), residue-type aware, instead of a flat divisor. A flat divisor
# (the old approach here) systematically inflates the apparent
# accessibility of bulky residues -- Trp/Tyr/Arg/Phe/Lys -- because their
# raw SASA is large even when substantially buried. Those are exactly the
# residue classes NES (Leu/Ile/Val/Phe-rich) and NLS (Lys/Arg-rich) motifs
# are built from, which is what was causing buried motifs to score as
# highly accessible.
MAX_ASA_TIEN2013 = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}
DEFAULT_MAX_ASA = 200.0  # generic fallback for non-standard/modified residues

RSA_FALLBACK_VALUE = 0.4  # Neutral mid-range RSA placeholder (NOT real exposure data)

def _rsa_from_raw(raw_by_key, resname_by_key, keys):
    """Convert a dict of raw per-residue SASA (Ų) into RSA (0-1ish) using
    residue-specific Tien et al. 2013 max ASA."""
    out = []
    for k in keys:
        max_asa = MAX_ASA_TIEN2013.get(resname_by_key[k], DEFAULT_MAX_ASA)
        out.append(min(raw_by_key.get(k, 0.0) / max_asa, 1.5))
    return out


def _zscores_np(values):
    """Population z-score (matches consensus_accessibility.py's statistics.pstdev
    convention, ddof=0) of a 1-D sequence of values across THIS chain's own
    distribution. Returns all zeros if stdev is 0 or fewer than 2 values --
    flags residues unusually buried/exposed relative to the rest of the
    protein, independent of any one method's absolute scale."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return [0.0] * len(arr)
    sd = float(arr.std())
    if sd == 0:
        return [0.0] * len(arr)
    mean = float(arr.mean())
    return ((arr - mean) / sd).tolist()


def calculate_sasa(structure, pdb_text=None, return_stats=False, return_raw=False):
    """
    Calculate per-residue RELATIVE solvent accessibility (RSA), 0 = fully
    buried .. ~1 = fully exposed, normalized against Tien et al. 2013
    residue-specific theoretical max ASA -- NOT a flat divisor (see
    consensus_accessibility.py for the full writeup of why that matters).

    If pdb_text (raw PDB file contents) is supplied, computes a 3-method
    consensus: FreeSASA Lee-Richards + FreeSASA Shrake-Rupley + Biopython
    Shrake-Rupley, each converted to RSA and averaged per residue. If
    pdb_text isn't given, falls back to Biopython Shrake-Rupley only
    (still Tien-normalized -- just single-method instead of consensus).

    Returns a tuple: (rsa_values, computed)
      - computed=True  -> rsa_values are real per-residue RSA
      - computed=False -> real calculation failed; rsa_values is a flat
                           neutral fallback and MUST NOT be treated as real
                           exposure data

    If return_stats=True, returns a 4-tuple instead:
    (rsa_values, computed, consensus_z, agreement_sd) -- ported from
    consensus_accessibility.py (previously a standalone diagnostic script,
    never wired into the live app):
      - consensus_z:  mean of each available method's z-score (RSA
                       standardized against that method's own distribution
                       across THIS chain, then averaged across methods) --
                       flags residues unusually buried/exposed relative to
                       the rest of this protein specifically, not just in
                       absolute RSA terms.
      - agreement_sd: stdev across the 3 methods per residue (low = the
                       methods agree, trustworthy; high = they disagree,
                       treat with caution). None per-residue when only one
                       method was available (nothing to compare it against).
    Existing callers that don't pass return_stats are completely unaffected
    -- default is the original 2-tuple.

    return_raw=True appends ONE MORE element after whatever
    return_stats already produces -- raw_sasa_values, the same 3-method
    consensus but WITHOUT the Tien normalization (i.e. real absolute Å²,
    not a 0-1 fraction). RSA is the right choice for "is this residue
    exposed" (removes the residue-size confound -- see the comment in
    nes_ml_predictor_improved.py's _extract_features about the bug this
    fixed), but it deliberately discards absolute magnitude: two residues
    at the same RSA can present very different absolute hydrophobic
    surface area depending on identity (e.g. exposed Trp vs exposed Ala).
    Absolute buried/exposed Å² is the physically relevant quantity for a
    CRM1 binding-affinity proxy specifically (buried hydrophobic surface
    area correlates with binding free energy, Chothia 1976), so this is
    additive here for that use only -- NOT wired into any ML training
    feature (default False, existing callers unaffected).
    """
    try:
        from Bio.PDB import ShrakeRupley

        sr = ShrakeRupley()
        sr.compute(structure, level="R")

        keys = []
        resnames = {}
        bio_raw = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':
                        key = (chain.id, residue.id[1])
                        keys.append(key)
                        resnames[key] = residue.resname
                        bio_raw[key] = residue.sasa
            break  # first model only

        rsa_bio = _rsa_from_raw(bio_raw, resnames, keys)

        if pdb_text:
            try:
                import freesasa, tempfile, os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                    f.write(pdb_text)
                    tmp_path = f.name
                try:
                    fs_struct = freesasa.Structure(tmp_path)
                    lr_result = freesasa.calc(fs_struct, freesasa.Parameters({'algorithm': freesasa.LeeRichards}))
                    sr_result = freesasa.calc(fs_struct, freesasa.Parameters({'algorithm': freesasa.ShrakeRupley}))

                    lr_raw, sr_raw = {}, {}
                    for chain_id, residues in lr_result.residueAreas().items():
                        for resnum_str, area in residues.items():
                            try:
                                lr_raw[(chain_id, int(resnum_str))] = area.total
                            except ValueError:
                                pass
                    for chain_id, residues in sr_result.residueAreas().items():
                        for resnum_str, area in residues.items():
                            try:
                                sr_raw[(chain_id, int(resnum_str))] = area.total
                            except ValueError:
                                pass

                    # fall back to the Biopython raw value for any residue
                    # FreeSASA didn't report (e.g. HETATM edge cases)
                    for k in keys:
                        lr_raw.setdefault(k, bio_raw[k])
                        sr_raw.setdefault(k, bio_raw[k])

                    rsa_lr = _rsa_from_raw(lr_raw, resnames, keys)
                    rsa_sr = _rsa_from_raw(sr_raw, resnames, keys)

                    consensus = [
                        (a + b + c) / 3.0
                        for a, b, c in zip(rsa_lr, rsa_sr, rsa_bio)
                    ]
                    # Raw (non-Tien-normalized) consensus, Å² -- same 3-method
                    # averaging as `consensus` above, just without the /max_ASA
                    # division. Only built when return_raw=True (default False).
                    raw_consensus = None
                    if return_raw:
                        raw_consensus = [
                            (lr_raw[k] + sr_raw[k] + bio_raw[k]) / 3.0 for k in keys
                        ]
                    if return_stats:
                        z_lr = _zscores_np(rsa_lr)
                        z_sr = _zscores_np(rsa_sr)
                        z_bio = _zscores_np(rsa_bio)
                        consensus_z = [(a + b + c) / 3.0 for a, b, c in zip(z_lr, z_sr, z_bio)]
                        agreement_sd = [float(np.std([a, b, c])) for a, b, c in zip(rsa_lr, rsa_sr, rsa_bio)]
                        if return_raw:
                            return consensus, True, consensus_z, agreement_sd, raw_consensus
                        return consensus, True, consensus_z, agreement_sd
                    if return_raw:
                        return consensus, True, raw_consensus
                    return consensus, True
                finally:
                    os.unlink(tmp_path)
            except Exception as e:
                print(f"   (FreeSASA consensus unavailable, using Biopython-only RSA: {e})")
                raw_bio = [bio_raw[k] for k in keys] if return_raw else None
                if return_stats:
                    z_bio = _zscores_np(rsa_bio)
                    # Only one method available -- z-score vs this chain's own
                    # distribution is still meaningful; agreement_sd has
                    # nothing to compare against, so it's reported as None.
                    if return_raw:
                        return rsa_bio, True, z_bio, [None] * len(rsa_bio), raw_bio
                    return rsa_bio, True, z_bio, [None] * len(rsa_bio)
                if return_raw:
                    return rsa_bio, True, raw_bio
                return rsa_bio, True

        raw_bio = [bio_raw[k] for k in keys] if return_raw else None
        if return_stats:
            z_bio = _zscores_np(rsa_bio)
            if return_raw:
                return rsa_bio, True, z_bio, [None] * len(rsa_bio), raw_bio
            return rsa_bio, True, z_bio, [None] * len(rsa_bio)
        if return_raw:
            return rsa_bio, True, raw_bio
        return rsa_bio, True
    except Exception as e:
        print(f"Warning: RSA calculation FAILED - using neutral fallback ({RSA_FALLBACK_VALUE}). "
              f"These are NOT real exposure values. Error: {e}")
        num_residues = sum(1 for model in structure for chain in model for residue in chain if residue.id[0] == ' ')
        if return_stats:
            if return_raw:
                return ([RSA_FALLBACK_VALUE] * num_residues, False,
                        [None] * num_residues, [None] * num_residues,
                        [None] * num_residues)
            return ([RSA_FALLBACK_VALUE] * num_residues, False,
                    [None] * num_residues, [None] * num_residues)
        if return_raw:
            return [RSA_FALLBACK_VALUE] * num_residues, False, [None] * num_residues
        return [RSA_FALLBACK_VALUE] * num_residues, False

def calculate_disorder_score(residue_names, plddt, sasa_values):
    """Calculate disorder score based on multiple factors"""
    disorder_prone = {
        'P': 0.9, 'E': 0.7, 'S': 0.7, 'Q': 0.6, 'K': 0.6,
        'A': 0.5, 'G': 0.5, 'D': 0.5, 'T': 0.4, 'R': 0.4,
        'N': 0.3, 'H': 0.2, 'M': 0.2, 'Y': 0.1, 'F': 0.1,
        'L': 0.1, 'I': 0.1, 'V': 0.1, 'W': 0.1, 'C': 0.1,
        'X': 0.5
    }

    n = len(residue_names)
    disorder_scores = []

    for i in range(n):
        plddt_score = max(0, (70 - plddt[i]) / 70)
        sasa_score = min(1.0, sasa_values[i])  # already RSA (Tien-normalized), no flat divisor
        aa_score = disorder_prone.get(residue_names[i], 0.5)

        window = 5
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        local_residues = residue_names[start:end]
        local_disorder = np.mean([disorder_prone.get(res, 0.5) for res in local_residues])

        disorder_score = (
            0.4 * plddt_score +
            0.2 * sasa_score +
            0.2 * aa_score +
            0.2 * local_disorder
        )

        disorder_scores.append(disorder_score)

    disorder_scores = gaussian_filter1d(disorder_scores, sigma=2).tolist()

    return disorder_scores


def calculate_flexibility_from_plddt(plddt, window_size=5):
    """
    Calculate local flexibility from pLDDT variance.
    Higher variance = more flexible = BETTER for NES

    Returns normalized flexibility scores (0-1)
    """
    plddt = np.array(plddt)
    n = len(plddt)
    flexibility = np.zeros(n)

    for i in range(n):
        # Get window around residue
        start = max(0, i - window_size//2)
        end = min(n, i + window_size//2 + 1)
        window = plddt[start:end]

        # Calculate variance (high variance = flexible)
        variance = np.var(window)

        # Also consider absolute pLDDT (lower = more flexible)
        avg_plddt = np.mean(window)

        # Combined flexibility score
        # High variance OR low pLDDT = flexible
        variance_score = min(1.0, variance / 400.0)  # Normalize variance
        plddt_score = 1.0 - (avg_plddt / 100.0)  # Invert pLDDT (low = flexible)

        # Combine both signals
        flexibility[i] = 0.6 * variance_score + 0.4 * plddt_score

    return flexibility.tolist()


def is_in_uniprot_disorder(residue_num, disorder_regions):
    """Check if a residue number falls within any UniProt disorder region"""
    for region in disorder_regions:
        if isinstance(region, dict):
            if 'start' in region and 'end' in region:
                if region['start'] <= residue_num <= region['end']:
                    return True
        elif len(region) == 2:  # Tuple format (start, end)
            if region[0] <= residue_num <= region[1]:
                return True
    return False


def predict_nes_motifs(sequence):
    """
    Predict Nuclear Export Signal (NES) motifs based on consensus patterns.
    NES consensus: Φ-X{2,3}-Φ-X{2,3}-Φ-X-Φ where Φ = hydrophobic (L, I, V, F, M)
    """
    import re

    # Hydrophobic residues that commonly appear in NES
    hydrophobic = ['L', 'I', 'V', 'F', 'M']

    # NES patterns with varying spacings
    # Pattern: Φ-X{2,3}-Φ-X{2,3}-Φ-X-Φ
    patterns = [
        r'[LIVFM].{2}[LIVFM].{2}[LIVFM].[LIVFM]',  # 2-2-1 spacing
        r'[LIVFM].{3}[LIVFM].{2}[LIVFM].[LIVFM]',  # 3-2-1 spacing
        r'[LIVFM].{2}[LIVFM].{3}[LIVFM].[LIVFM]',  # 2-3-1 spacing
        r'[LIVFM].{3}[LIVFM].{3}[LIVFM].[LIVFM]',  # 3-3-1 spacing
    ]

    nes_matches = []

    for pattern in patterns:
        for match in re.finditer(pattern, sequence):
            start = match.start()
            end = match.end()
            motif_seq = match.group()

            # Validate length (NES are 8-15 residues)
            if len(motif_seq) < 8 or len(motif_seq) > 15:
                continue

            # Count hydrophobic residues
            hydrophobic_count = sum(1 for aa in motif_seq if aa in hydrophobic)

            # Require minimum 4 hydrophobic residues (critical for CRM1)
            if hydrophobic_count < 4:
                continue

            # Calculate motif score based on hydrophobic content
            score = hydrophobic_count / len(motif_seq)

            nes_matches.append({
                'start': start + 1,  # 1-indexed
                'end': end,
                'sequence': motif_seq,
                'score': score,
                'pattern': pattern
            })

    # Remove overlapping matches, keeping higher scores
    nes_matches.sort(key=lambda x: x['score'], reverse=True)
    non_overlapping = []

    for match in nes_matches:
        overlaps = False
        for existing in non_overlapping:
            if not (match['end'] < existing['start'] or match['start'] > existing['end']):
                overlaps = True
                break
        if not overlaps:
            non_overlapping.append(match)

    return non_overlapping


def predict_nes_motifs_flexible(sequence):
    """
    FLEXIBLE NES prediction with length range 6-17 amino acids
    Optimal: 8-15 aa
    Acceptable: 6-7 aa (penalized) or 16-17 aa (penalized)
    Cutoff: <6 aa or >17 aa (rejected)
    """
    import re

    # Hydrophobic residues that commonly appear in NES
    hydrophobic = ['L', 'I', 'V', 'F', 'M']

    # NES patterns with varying spacings
    patterns = [
        r'[LIVFM].{2}[LIVFM].{2}[LIVFM].[LIVFM]',  # 2-2-1 spacing (8 aa)
        r'[LIVFM].{3}[LIVFM].{2}[LIVFM].[LIVFM]',  # 3-2-1 spacing (9 aa)
        r'[LIVFM].{2}[LIVFM].{3}[LIVFM].[LIVFM]',  # 2-3-1 spacing (9 aa)
        r'[LIVFM].{3}[LIVFM].{3}[LIVFM].[LIVFM]',  # 3-3-1 spacing (10 aa)
        r'[LIVFM].{1}[LIVFM].{2}[LIVFM].[LIVFM]',  # 1-2-1 spacing (6 aa) - SHORT
        r'[LIVFM].{2}[LIVFM].{2}[LIVFM].{2}[LIVFM]', # 2-2-2 spacing (9 aa)
        r'[LIVFM].{4}[LIVFM].{3}[LIVFM].[LIVFM]',  # 4-3-1 spacing (11 aa) - LONG
    ]

    nes_matches = []

    for pattern in patterns:
        for match in re.finditer(pattern, sequence):
            start = match.start()
            end = match.end()
            motif_seq = match.group()
            motif_len = len(motif_seq)

            # HARD CUTOFF: Reject if < 6 or > 17
            if motif_len < 6 or motif_len > 17:
                continue

            # Count hydrophobic residues
            hydrophobic_count = sum(1 for aa in motif_seq if aa in hydrophobic)

            # Require minimum 4 hydrophobic residues
            if hydrophobic_count < 4:
                continue

            # Base score from hydrophobic content
            base_score = hydrophobic_count / motif_len

            # LENGTH PENALTY/BONUS
            if 8 <= motif_len <= 15:
                # OPTIMAL length - no penalty
                length_factor = 1.0
            elif 6 <= motif_len < 8:
                # Too short - penalize progressively
                # 7 aa: 0.9x, 6 aa: 0.8x
                length_factor = 0.7 + (motif_len - 6) * 0.15
            elif 15 < motif_len <= 17:
                # Too long - penalize slightly
                # 16 aa: 0.95x, 17 aa: 0.9x
                length_factor = 1.0 - (motif_len - 15) * 0.05
            else:
                # Should never reach here due to hard cutoff
                continue

            # Final score with length adjustment
            final_score = base_score * length_factor

            nes_matches.append({
                'start': start + 1,  # 1-indexed
                'end': end,
                'sequence': motif_seq,
                'score': round(final_score, 3),
                'length': motif_len,
                'length_factor': round(length_factor, 2),
                'pattern': pattern
            })

    # Remove overlapping matches, keeping higher scores
    nes_matches.sort(key=lambda x: x['score'], reverse=True)
    non_overlapping = []

    for match in nes_matches:
        overlaps = False
        for existing in non_overlapping:
            if not (match['end'] < existing['start'] or match['start'] > existing['end']):
                overlaps = True
                break
        if not overlaps:
            non_overlapping.append(match)

    return non_overlapping


def calculate_crm1_binding_score(residue_names, residue_numbers, plddt, sasa_values,
                                  hydrophobicity, disorder_scores, nes_motifs, sequence=None):
    """
    Calculate CRM1 binding likelihood integrating sequence and structural features.
    NOW ENHANCED: Automatically uses ML predictions if available!
    Based on principles from CRM1-NES binding literature.
    """
    n = len(residue_names)
    crm1_scores = np.zeros(n)

    # Try to get ML predictions if available
    ml_scores = np.zeros(n)
    ml_available = False

    if ml_predictor is not None and sequence is not None:
        try:
            print("  [ML] Running ML-enhanced predictions...")
            window_size = 10
            ml_count = 0
            for i in range(len(sequence) - window_size + 1):
                subseq = sequence[i:i+window_size]

                # Get pLDDT and SASA for this window
                window_plddt = plddt[i:i+window_size] if i+window_size <= len(plddt) else None
                window_sasa = sasa_values[i:i+window_size] if i+window_size <= len(sasa_values) else None

                # ML prediction
                # This function (calculate_crm1_binding_score) is not
                # called anywhere else in app.py -- confirmed via search, it's dead
                # code from an earlier version, superseded by the unified STEP 3
                # scanning pipeline. It's left in place (not deleted) rather than
                # erased, but the predict() call below previously omitted
                # full_sequence/nes_start entirely, meaning EVERY window here got
                # zero flanking analysis (worse than the terminus-only truncation
                # issue fixed elsewhere) -- now passing them so this path is
                # consistent with the real pipeline if it's ever revived.
                try:
                    result = ml_predictor.predict(subseq, full_sequence=sequence,
                                                   nes_start=i,
                                                   plddt=window_plddt, sasa=window_sasa)
                    # Handle both old (2 values) and new (3 values) return format
                    if len(result) == 3:
                        ml_prob, ml_conf, ml_details = result
                    else:
                        ml_prob, ml_conf = result

                    # Apply ML score to this window
                    for j in range(i, min(i+window_size, n)):
                        ml_scores[j] = max(ml_scores[j], ml_prob)

                    if ml_prob > 0.6:
                        ml_count += 1
                except:
                    pass

            ml_available = True
            print(f"  [ML] Integrated ML predictions: {ml_count} high-confidence regions found")
        except Exception as e:
            print(f"  [ML] Could not apply ML predictions: {e}")

    # For each NES motif, calculate enhanced score with structural context
    for motif in nes_motifs:
        start_idx = motif['start'] - 1  # Convert to 0-indexed
        end_idx = motif['end'] - 1

        if start_idx < 0 or end_idx >= n:
            continue

        # Expand window slightly for context
        window_start = max(0, start_idx - 2)
        window_end = min(n, end_idx + 3)

        # Extract features for this region
        region_sasa = sasa_values[window_start:window_end]
        region_hydro = hydrophobicity[window_start:window_end]
        region_plddt = plddt[window_start:window_end]
        region_disorder = disorder_scores[window_start:window_end]
        region_ml = ml_scores[window_start:window_end] if ml_available else []

        # Calculate structural suitability
        # 1. Surface accessibility (NES must be exposed)
        avg_sasa = np.mean(region_sasa)
        sasa_score = min(1.0, avg_sasa)  # already RSA (Tien-normalized), no flat divisor

        # 2. Hydrophobicity (NES should be hydrophobic)
        avg_hydro = np.mean(region_hydro)
        hydro_score = (avg_hydro + 4.5) / 9.0  # Normalize Kyte-Doolittle scale
        hydro_score = max(0, min(1, hydro_score))

        # 3. Confidence (prefer high confidence regions)
        avg_plddt = np.mean(region_plddt)
        confidence_score = avg_plddt / 100.0

        # 4. Flexibility (NES often in loops/flexible regions)
        avg_disorder = np.mean(region_disorder)
        flexibility_score = avg_disorder  # Already 0-1

        # 5. Sequence motif score
        motif_score = motif['score']

        # 6. ML score (if available)
        avg_ml = np.mean(region_ml) if len(region_ml) > 0 and ml_available else 0

        # Integrated CRM1 binding score - ML-ENHANCED if available!
        if ml_available and avg_ml > 0:
            # ML-ENHANCED scoring
            integrated_score = (
                0.15 * motif_score +        # Sequence pattern
                0.20 * sasa_score +          # Surface exposure
                0.15 * hydro_score +         # Hydrophobicity
                0.10 * confidence_score +    # Structural confidence
                0.10 * flexibility_score +   # Local flexibility
                0.30 * avg_ml                # ML prediction (HIGHEST WEIGHT!)
            )
        else:
            # Traditional scoring without ML
            integrated_score = (
                0.25 * motif_score +        # Sequence pattern
                0.25 * sasa_score +          # Surface exposure
                0.20 * hydro_score +         # Hydrophobicity
                0.15 * confidence_score +    # Structural confidence
                0.15 * flexibility_score     # Local flexibility
            )

        # Apply score to region with Gaussian smoothing
        for i in range(window_start, window_end):
            distance_factor = 1.0 - abs(i - (start_idx + end_idx) / 2) / ((window_end - window_start) / 2)
            distance_factor = max(0, distance_factor)
            crm1_scores[i] = max(crm1_scores[i], integrated_score * distance_factor)

    # If ML available but no motifs found, still consider ML scores
    if ml_available and len(nes_motifs) == 0 and np.max(ml_scores) > 0.6:
        print("  [ML] No traditional motifs found, but ML detected high-confidence regions")
        crm1_scores = ml_scores * 0.5  # Use ML scores but scaled down without motif confirmation

    # Smooth the scores
    crm1_scores = gaussian_filter1d(crm1_scores, sigma=1.5)

    return crm1_scores.tolist()


@app.route('/api/crm1_analysis/<model_id>', methods=['GET'])
def analyze_crm1_binding(model_id):
    """
    ENHANCED CRM1/NES Analysis with:
    - ML prediction (25%)
    - fpocket CRM1 binding (30%) - HIGHEST WEIGHT!
    - Hydrophobicity (15%)
    - SASA (10%)
    - Flexibility from pLDDT variance (10%)
    - Disorder (AlphaFold + UniProt) (10%)
    """
    try:
        print(f"\n{'='*70}")
        print(f"ENHANCED CRM1/NES ANALYSIS: {model_id}")
        print(f"{'='*70}")

        # Download structure
        pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
        response = requests.get(pdb_url, timeout=30)

        if response.status_code != 200:
            return jsonify({'error': 'Could not fetch structure'}), 404

        structure_content = response.text

        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', StringIO(structure_content))

        # Extract sequence and features
        residue_names = []
        residue_numbers = []
        plddt = []
        coords = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':
                        for atom in residue:
                            if atom.name == 'CA':
                                coords.append(atom.coord.tolist())
                                plddt.append(atom.bfactor)

                                try:
                                    res_name = three_to_one(residue.resname)
                                except:
                                    res_name = 'X'

                                residue_names.append(res_name)
                                residue_numbers.append(residue.id[1])

        sequence = ''.join(residue_names)
        print(f"  Sequence length: {len(sequence)} residues")

        # =====================================================================
        # STEP 1: Calculate ALL structural features
        # =====================================================================
        print("\n[1/6] Calculating structural features...")
        sasa_values, sasa_computed = calculate_sasa(structure, pdb_text=structure_content)
        hydrophobicity = [HYDROPHOBICITY.get(res, 0.0) for res in residue_names]
        disorder_scores = calculate_disorder_score(residue_names, plddt, sasa_values)
        flexibility_scores = calculate_flexibility_from_plddt(plddt, window_size=5)

        print(f"  Accessibility (RSA) calculated" if sasa_computed else "  Warning: RSA fallback used (not real exposure data)")
        print(f"  Hydrophobicity calculated")
        print(f"  Disorder calculated")
        print(f"  Flexibility calculated (from pLDDT variance)")

        # =====================================================================
        # STEP 2: Fetch UniProt disorder regions
        # =====================================================================
        print("\n[2/6] Fetching UniProt disorder annotations...")
        # Extract UniProt ID from model_id (e.g., AF-P12345-F1 -> P12345)
        uniprot_id = model_id.split('-')[1] if '-' in model_id else model_id
        uniprot_disorder_regions = fetch_uniprot_disorder_regions(uniprot_id)
        print(f"  Found {len(uniprot_disorder_regions)} UniProt disorder regions")

        # IUPred2A/ANCHOR2: real sequence-based disorder, now PRIMARY signal
        # in place of the calculate_disorder_score() pLDDT/composition
        # heuristic wherever it's available (heuristic stays as fallback --
        # see calculate_enhanced_disorder_score's docstring for the full
        # rationale). Same per-residue array shape as disorder_scores
        # already had, so every downstream window/flank slice below keeps
        # working unchanged.
        anchor2_scores = None
        iupred_raw, anchor2_raw, iupred_seq = fetch_iupred2a_scores(uniprot_id)
        iupred_aligned, anchor2_aligned = align_iupred_to_structure(sequence, iupred_seq, iupred_raw, anchor2_raw)
        if iupred_aligned:
            disorder_scores = iupred_aligned
            anchor2_scores = anchor2_aligned
            print(f"  Using IUPred2A disorder ({len(iupred_aligned)} residues)"
                  + (" + ANCHOR2 binding-region scores" if anchor2_aligned else ""))
        else:
            print(f"  Warning: IUPred2A unavailable -- using structural heuristic disorder")

        # =====================================================================
        # STEP 3: Run fpocket analysis for CRM1 binding pockets
        # =====================================================================
        print("\n[3/6] Detecting CRM1-compatible pockets (fpocket)...")
        pocket_scores = np.zeros(len(sequence))

        if pocket_detector is not None:
            try:
                pockets = pocket_detector.detect_pockets(structure_content)
                print(f"  Found {len(pockets)} CRM1-compatible pockets")

                # Map pocket scores to residues
                for pocket in pockets:
                    for res_num in pocket.get('residue_numbers', []):
                        if res_num in residue_numbers:
                            idx = residue_numbers.index(res_num)
                            # Use CRM1 compatibility score
                            pocket_scores[idx] = max(pocket_scores[idx],
                                                   pocket.get('crm1_compatibility_score', 0))
            except Exception as e:
                print(f"  Warning: fpocket failed: {e}")
        else:
            print("  Warning: fpocket not available")

        # =====================================================================
        # STEP 4: ML predictions
        # =====================================================================
        print("\n[4/6] Running ML-based NES predictions...")
        ml_scores = np.zeros(len(sequence))

        if ml_predictor is not None:
            try:
                window_size = 10
                ml_count = 0

                for i in range(len(sequence) - window_size + 1):
                    subseq = sequence[i:i+window_size]
                    window_plddt = plddt[i:i+window_size]
                    window_sasa = sasa_values[i:i+window_size]

                    result = ml_predictor.predict(subseq, plddt=window_plddt, sasa=window_sasa)
                    # Handle both old (2 values) and new (3 values) return format
                    if len(result) == 3:
                        ml_prob, ml_conf, ml_details = result
                    else:
                        ml_prob, ml_conf = result

                    # Apply to window
                    for j in range(i, min(i+window_size, len(sequence))):
                        ml_scores[j] = max(ml_scores[j], ml_prob)

                    if ml_prob > 0.6:
                        ml_count += 1

                print(f"  ML analysis complete: {ml_count} high-confidence windows")
            except Exception as e:
                print(f"  Warning: ML prediction failed: {e}")
        else:
            print("  Warning: ML predictor not available")

        # =====================================================================
        # STEP 5: Pattern-based NES motif detection
        # =====================================================================
        print("\n[5/6] Detecting NES motifs by pattern...")
        nes_motifs = predict_nes_motifs_flexible(sequence)
        print(f"  Found {len(nes_motifs)} pattern-based NES motifs")

        # =====================================================================
        # STEP 6: COMPREHENSIVE SCORING - PROPERLY WEIGHTED!
        # =====================================================================
        print("\n[6/6] Calculating comprehensive NES scores...")

        # Initialize arrays
        comprehensive_scores = []

        # Scan with sliding window
        window_size = 11  # Standard NES length

        for i in range(len(sequence) - window_size + 1):
            window_end = i + window_size

            # Extract features for this window
            window_sasa = np.mean(sasa_values[i:window_end])
            window_disorder = np.mean(disorder_scores[i:window_end])
            window_flexibility = np.mean(flexibility_scores[i:window_end])
            window_ml = np.mean(ml_scores[i:window_end])
            window_pocket = np.mean(pocket_scores[i:window_end])

            # FLANKING REGION ANALYSIS - Key for accessibility!
            # Check SASA and disorder in flanking regions (±5 residues)
            flank_size = 5
            n_flank_start = max(0, i - flank_size)
            c_flank_end = min(len(sequence), window_end + flank_size)

            # N-terminal flanking accessibility
            n_flank_sasa = np.mean(sasa_values[n_flank_start:i]) if i > 0 else window_sasa
            n_flank_disorder = np.mean(disorder_scores[n_flank_start:i]) if i > 0 else window_disorder

            # C-terminal flanking accessibility
            c_flank_sasa = np.mean(sasa_values[window_end:c_flank_end]) if window_end < len(sequence) else window_sasa
            c_flank_disorder = np.mean(disorder_scores[window_end:c_flank_end]) if window_end < len(sequence) else window_disorder

            # Combined flanking accessibility score (both terms already RSA, 0-1)
            flanking_accessibility = (n_flank_sasa + c_flank_sasa) / 2.0
            flanking_disorder = (n_flank_disorder + c_flank_disorder) / 2.0

            # Check if in UniProt disorder region
            mid_residue = residue_numbers[i + window_size//2]
            in_uniprot_disorder = is_in_uniprot_disorder(mid_residue, uniprot_disorder_regions)

            # Check if overlaps with pattern motif
            overlaps_motif = False
            motif_class = None
            motif_score = 0.0
            for motif in nes_motifs:
                if not (window_end < motif['start'] or i + 1 > motif['end']):
                    overlaps_motif = True
                    motif_class = motif.get('class', 'unknown')
                    motif_score = max(motif_score, motif['score'])

            # Normalize individual scores (0-1)
            sasa_norm = min(1.0, window_sasa)  # already RSA (Tien-normalized), no flat divisor
            disorder_norm = window_disorder  # Already 0-1
            flexibility_norm = window_flexibility  # Already 0-1
            ml_norm = window_ml  # Already 0-1
            pocket_norm = window_pocket  # Already 0-1
            flanking_access_norm = min(1.0, flanking_accessibility)
            flanking_disorder_norm = flanking_disorder
            window_anchor2 = float(np.mean(anchor2_scores[i:window_end])) if anchor2_scores is not None else None

            # Enhanced disorder score (AlphaFold + UniProt + flanking)
            combined_disorder = disorder_norm
            # UniProt-annotation bonus KEPT at full strength: this is
            # genuinely independent evidence the trained model never sees --
            # its 20 (NLS) / 37 (NES) input features are all computed
            # quantities (AlphaFold pLDDT-derived disorder, sequence
            # composition, etc.), never a curated UniProt IDR annotation. A
            # real database annotation of "this region is disordered" is a
            # different, higher-quality kind of evidence than anything
            # ml_norm already incorporates.
            if in_uniprot_disorder:
                combined_disorder = min(1.0, disorder_norm * 1.3)  # 30% bonus for UniProt annotation

            # REMOVED: a flanking-disorder bonus (most recently 1.05x, down
            # from an original 1.2x) used to live here. Unlike the UniProt
            # bonus above, n_flank_disorder/c_flank_disorder ARE literal
            # input features of the trained NES model -- so on top of
            # already being weak evidence on their own (c_flank_disorder
            # point-biserial p=0.68, not significant; n_flank_disorder
            # r=-0.09), boosting combined_disorder for it was doubly
            # redundant: ml_norm already reflects whatever real signal these
            # two features carry, and combined_disorder's own weight in
            # base_score below is now small for the same reason.

            # ============================================================
            # DATA-DRIVEN SCORING (see NES_COMPREHENSIVE_WEIGHTS above).
            # ml_norm is now the dominant term (0.75): it's the trained
            # model's actual output probability, already computed from
            # sasa_norm, nes_disorder_mean, and plddt_norm directly (all 3
            # are literal model input features), so sasa_norm/
            # combined_disorder/flexibility_norm below would be re-adding
            # evidence the model already weighed in if kept at their old
            # weights -- they're now small, diagnostics-proportioned
            # nudges rather than co-equal votes. flanking_access_norm stays
            # meaningful (0.15) because flank *SASA* was never a model
            # feature (only flank disorder was), so it's the one term here
            # that's still genuinely independent information.
            # ============================================================
            base_score = (
                NES_COMPREHENSIVE_WEIGHTS['ml_norm'] * ml_norm +
                NES_COMPREHENSIVE_WEIGHTS['sasa_norm'] * sasa_norm +
                NES_COMPREHENSIVE_WEIGHTS['combined_disorder'] * combined_disorder +
                NES_COMPREHENSIVE_WEIGHTS['flanking_access_norm'] * flanking_access_norm +
                NES_COMPREHENSIVE_WEIGHTS['flexibility_norm'] * flexibility_norm
            )
            # Note: Hydrophobicity REMOVED - already in ML model and patterns!

            # ANCHOR2 bonus: NES motifs are short linear binding elements
            # sitting in disordered regions -- exactly what ANCHOR2 predicts
            # (disordered *binding* region probability, from the same
            # IUPred2A call as disorder_scores above). Kept as a fixed,
            # hand-picked additive nudge rather than folded into
            # NES_COMPREHENSIVE_WEIGHTS's data-derived dict: ANCHOR2 was
            # never a trained-model feature, so there's nothing in
            # diagnosis_report.json to derive a weight from (same reasoning
            # as flanking_access_norm's fixed weight above). Modest by
            # design until there's real validation data to size it properly.
            if window_anchor2 is not None:
                base_score = min(1.0, base_score + window_anchor2 * 0.08)

            # PATTERN MATCH bonus, SHRUNK (50%/10% -> 15%/5%): overlaps_motif
            # comes from predict_nes_motifs_flexible(), a plain hydrophobic-
            # spacing regex scan -- conceptually the same "leucine register"
            # signal ml_norm already carries via pssm_score, just a second,
            # independently-coded implementation of the same idea rather
            # than genuinely new evidence. A 50% multiplicative boost for
            # something ml_norm mostly already reflects was the single
            # largest remaining piece of the pattern/ML double-counting
            # problem -- shrunk rather than removed outright since it's a
            # literature-grounded heuristic and a bigger behavioral change
            # than the other trims here; flag if this still looks too
            # strong (or too weak) once you see real predictions with it.
            if overlaps_motif:
                base_score = min(1.0, base_score * 1.15)  # DOWN from 1.5
                if motif_class in ['class_1a', 'class_1b']:
                    base_score = min(1.0, base_score * 1.05)  # DOWN from 1.1

            # Pocket provides VALIDATION BONUS (not requirement)
            if pocket_norm > 0.7:
                combined_score = min(1.0, base_score * 1.3)  # 30% bonus for strong pocket
            elif pocket_norm > 0.4:
                combined_score = min(1.0, base_score * 1.15)  # 15% bonus for moderate pocket
            else:
                combined_score = base_score  # No penalty for no pocket - many real NES lack detectable pockets

            comprehensive_scores.append({
                'start': i + 1,
                'end': window_end,
                'sequence': sequence[i:window_end],
                'length': window_size,
                'combined_score': round(combined_score, 3),
                'base_score': round(base_score, 3),
                'pattern_match': overlaps_motif,
                'pattern_class': motif_class if overlaps_motif else None,
                'components': {
                    'ml_probability': round(ml_norm, 3),
                    'pocket_compatibility': round(pocket_norm, 3),
                    'surface_accessibility': round(sasa_norm, 3),
                    'flanking_accessibility': round(flanking_access_norm, 3),
                    'flexibility': round(flexibility_norm, 3),
                    'disorder': round(disorder_norm, 3),
                    'combined_disorder': round(combined_disorder, 3),
                    'flanking_disorder': round(flanking_disorder_norm, 3),
                    'uniprot_disorder': in_uniprot_disorder,
                    'anchor2_binding': round(window_anchor2, 3) if window_anchor2 is not None else None,
                    'disorder_source': 'iupred2a' if iupred_aligned else 'structural_heuristic'
                }
            })

        # Filter to high-scoring regions - LOWERED THRESHOLD!
        # Was 0.5, now 0.35 to catch more candidates
        filtered_regions = [r for r in comprehensive_scores if r['combined_score'] > 0.35]

        # Remove overlapping regions
        filtered_regions.sort(key=lambda x: x['combined_score'], reverse=True)
        non_overlapping = []

        for region in filtered_regions:
            overlaps = False
            for existing in non_overlapping:
                if not (region['end'] < existing['start'] or region['start'] > existing['end']):
                    overlaps = True
                    break
            if not overlaps:
                non_overlapping.append(region)

        # Count confidence levels (adjusted for new scoring)
        high_conf = sum(1 for r in non_overlapping if r['combined_score'] > 0.60)  # Lowered from 0.70
        medium_conf = sum(1 for r in non_overlapping if 0.45 < r['combined_score'] <= 0.60)  # Lowered from 0.55-0.70
        low_conf = sum(1 for r in non_overlapping if 0.35 < r['combined_score'] <= 0.45)  # New category

        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY (ML & Pattern-Focused Scoring)")
        print(f"{'='*70}")
        print(f"  Total predictions: {len(non_overlapping)}")
        print(f"  High confidence (>0.60): {high_conf}")
        print(f"  Medium confidence (0.45-0.60): {medium_conf}")
        print(f"  Low confidence (0.35-0.45): {low_conf}")

        # Show top 5 predictions with details
        if len(non_overlapping) > 0:
            print(f"\n{'='*70}")
            print(f"TOP PREDICTIONS (showing up to 5)")
            print(f"{'='*70}")
            for i, region in enumerate(non_overlapping[:5]):
                comp = region['components']
                print(f"\n{i+1}. {region['sequence']} (pos {region['start']}-{region['end']})")
                print(f"   Combined Score: {region['combined_score']:.3f} | Base: {region.get('base_score', 'N/A'):.3f}")
                if region.get('pattern_match'):
                    print(f"   Pattern Match: {region.get('pattern_class', 'unknown')}")
                print(f"   ML: {comp['ml_probability']:.2f} | SASA: {comp['surface_accessibility']:.2f} | "
                      f"Disorder: {comp['combined_disorder']:.2f} | Flex: {comp['flexibility']:.2f}")
                print(f"   Flanking: SASA={comp['flanking_accessibility']:.2f} Disorder={comp['flanking_disorder']:.2f}")
                if comp['pocket_compatibility'] > 0.3:
                    print(f"   Pocket: {comp['pocket_compatibility']:.2f} (bonus applied)")
                if comp['uniprot_disorder']:
                    print(f"   In UniProt disorder region")

        print(f"{'='*70}\n")

        return jsonify({
            'binding_regions': non_overlapping,
            'summary': {
                'filtered_predictions': len(non_overlapping),
                'high_confidence': high_conf,
                'medium_confidence': medium_conf,
                'low_confidence': low_conf,
                'pockets_detected': int(np.sum(pocket_scores > 0)),
                'sasa_computed': sasa_computed  # False = fallback placeholder used, not real exposure data
            }
        })

    except Exception as e:
        print(f"Error in enhanced CRM1 analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/domains/<protein_id>', methods=['GET'])
def get_domains(protein_id):
    """Get domain information from UniProt"""
    try:
        print(f"\n{'='*60}")
        print(f"Fetching domain information for: {protein_id}")
        print(f"{'='*60}")

        # Query UniProt API for protein features
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"

        response = requests.get(uniprot_url, timeout=10)

        if response.status_code != 200:
            print(f"Failed: UniProt API returned status {response.status_code}")
            return jsonify([])

        data = response.json()

        # Extract domain information from features
        domains = []
        domain_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
            '#FF8C94', '#A8E6CF', '#FFD3B6', '#FFAAA5', '#C7CEEA'
        ]

        features = data.get('features', [])
        domain_idx = 0

        for feature in features:
            feature_type = feature.get('type', '')

            # Look for domain-related features
            if feature_type in ['Domain', 'Region', 'Repeat', 'Zinc finger', 'DNA binding']:
                location = feature.get('location', {})
                start = location.get('start', {}).get('value')
                end = location.get('end', {}).get('value')

                if start and end:
                    description = feature.get('description', 'Unknown domain')

                    # Assign a color from the palette
                    color = domain_colors[domain_idx % len(domain_colors)]
                    domain_idx += 1

                    domains.append({
                        'type': feature_type,
                        'description': description,
                        'start': start,
                        'end': end,
                        'color': color
                    })

                    print(f"  Found {feature_type}: {description} ({start}-{end})")

        print(f"\nTotal domains found: {len(domains)}")
        print(f"{'='*60}\n")

        return jsonify(domains)

    except Exception as e:
        print(f"Error fetching domains: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])

@app.route('/api/analyze_region', methods=['POST'])
def analyze_region():
    """Analyze a specific region for PPI likelihood"""
    try:
        data = request.json
        start = int(data.get('start', 1))
        end = int(data.get('end', 1))
        sasa = data.get('sasa', [])
        charges = data.get('charges', [])
        disorder = data.get('disorder', [])
        hydrophobicity = data.get('hydrophobicity', [])
        sequence = data.get('sequence', '')  # full protein sequence, optional

        start_idx = start - 1
        end_idx = end

        region_sasa = sasa[start_idx:end_idx]
        region_charges = charges[start_idx:end_idx]
        region_disorder = disorder[start_idx:end_idx]
        region_hydro = hydrophobicity[start_idx:end_idx]

        # Real localCIDER linear hydropathy/NCPR/FCR profiles for the
        # selected region -- this is the CIDER-style graph data (matching
        # the CIDER webserver's "Linear hydropathy" / "Linear net charge per
        # residue" / "Linear fraction of charge per residue" plots), one
        # value per residue in the selected region, so the frontend can plot
        # them directly.
        cider_region_profile = None
        if sequence:
            region_seq = sequence[start_idx:end_idx]
            cider_region_profile = compute_linear_cider_profiles(region_seq)
            cider_region_profile['positions'] = list(range(start, start + len(region_seq)))

        avg_sasa = np.mean(region_sasa)
        total_sasa = np.sum(region_sasa)
        avg_charge = np.mean([abs(c) for c in region_charges])
        net_charge = sum(region_charges)
        avg_disorder = np.mean(region_disorder)
        avg_hydro = np.mean(region_hydro)

        ppi_score = 0.0

        sasa_percent = avg_sasa * 100  # avg_sasa is already RSA (0-1, Tien-normalized)
        if 30 <= sasa_percent <= 70:
            ppi_score += 0.4
        elif sasa_percent > 70:
            ppi_score += 0.2

        if avg_charge > 0.3:
            ppi_score += 0.2

        if avg_disorder < 0.3:
            ppi_score += 0.2
        elif avg_disorder < 0.5:
            ppi_score += 0.1

        if -0.5 < avg_hydro < 1.5:
            ppi_score += 0.2

        ppi_likelihood = min(100, ppi_score * 100)

        if ppi_likelihood >= 70:
            interpretation = "High likelihood of protein-protein interaction"
        elif ppi_likelihood >= 50:
            interpretation = "Moderate likelihood of protein-protein interaction"
        elif ppi_likelihood >= 30:
            interpretation = "Low to moderate likelihood of protein-protein interaction"
        else:
            interpretation = "Low likelihood of protein-protein interaction"

        return jsonify({
            'avg_sasa': round(avg_sasa, 2),
            'total_sasa': round(total_sasa, 2),
            'avg_charge': round(avg_charge, 3),
            'net_charge': round(net_charge, 3),
            'avg_disorder': round(avg_disorder, 3),
            'avg_hydrophobicity': round(avg_hydro, 3),
            'ppi_likelihood': round(ppi_likelihood, 1),
            'interpretation': interpretation,
            'cider_profile': cider_region_profile,  # None if no sequence was sent
        })

    except Exception as e:
        print(f"Error analyzing region: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ENHANCED ENDPOINTS
# ============================================================================

@app.route('/api/pocket_analysis/<model_id>', methods=['GET'])
def analyze_pockets(model_id):
    """
    NEW: Detect CRM1-binding pockets using fpocket
    """
    try:
        print(f"\n{'='*60}")
        print(f"POCKET ANALYSIS: {model_id}")
        print(f"{'='*60}")

        # Download PDB structure
        pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
        response = requests.get(pdb_url, timeout=30)

        if response.status_code != 200:
            return jsonify({'error': 'Could not download structure'}), 404

        pdb_content = response.text

        # Run pocket detection
        pockets = pocket_detector.detect_pockets(pdb_content)

        print(f"Found {len(pockets)} CRM1-compatible pockets")

        return jsonify({
            'model_id': model_id,
            'pockets': pockets,
            'total_pockets': len(pockets)
        })

    except Exception as e:
        print(f"Error in pocket analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml_predict', methods=['POST'])
def ml_predict_nes():
    """
    NEW: ML-based NES prediction
    """
    try:
        data = request.json
        sequence = data.get('sequence', '')
        plddt = data.get('plddt', None)
        sasa = data.get('sasa', None)

        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400

        # Run ML prediction with enhanced details
        result = ml_predictor.predict(sequence, plddt=plddt, sasa=sasa)
        # Handle both old (2 values) and new (3 values) return format
        if len(result) == 3:
            probability, confidence, details = result
        else:
            probability, confidence = result
            details = {}

        # Get feature importance
        feature_importance = {}
        if hasattr(ml_predictor, 'get_feature_importance'):
            feature_importance = ml_predictor.get_feature_importance()

        return jsonify({
            'probability': float(probability),
            'confidence': confidence,
            'details': details,  # Enhanced details from improved predictor
            'feature_importance': feature_importance
        })
        feature_importance = ml_predictor.get_feature_importance()

        return jsonify({
            'sequence': sequence,
            'nes_probability': round(probability, 3),
            'confidence': confidence,
            'feature_importance': feature_importance
        })

    except Exception as e:
        print(f"Error in ML prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/nls_predict', methods=['POST', 'OPTIONS'])
def nls_predict():
    """
    ML-based NLS (nuclear localization signal) prediction. Mirrors
    /api/ml_predict's request/response shape for the NES side, but calls
    the separate NLSPredictor (nls_ml_predictor.py) -- classical/bipartite
    basic-cluster PSSM + charge/hydrophobicity/flanking/disorder/CIDER
    features, not the NES model.

    plddt/sasa (optional): real per-residue arrays for `sequence` itself
    (same convention as /api/ml_predict on the NES side -- already local to
    the candidate, not sliced from a larger array here). Previously this
    endpoint accepted full_sequence/start/end for real flanking-region
    features but never accepted plddt/sasa at all, so sasa_norm silently
    used NLSPredictor._extract_features()'s neutral 0.50 default every time
    this endpoint was called, even when the caller had real structural data
    available (e.g. from a prior /api/analyze_alphafold load). /api/nls_scan
    (the whole-structure scan) always had real sasa wired through; this
    single-candidate endpoint didn't.
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        if nls_predictor is None or nls_predictor.model is None:
            return jsonify({'error': 'NLS predictor not available -- run '
                             '`python nls_ml_predictor.py train` in the AlphaFold '
                             'directory to train and save a model first.'}), 503

        data = request.json or {}
        sequence = data.get('sequence', '')
        full_sequence = data.get('full_sequence', None)
        start = data.get('start', None)
        end = data.get('end', None)
        plddt = data.get('plddt', None)
        sasa = data.get('sasa', None)

        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400

        result = nls_predictor.predict(sequence, full_sequence=full_sequence, start=start, end=end,
                                        plddt_values=plddt, sasa_values=sasa)
        feature_importance = nls_predictor.get_feature_importance()

        # Same accessibility gate as /api/nls_scan (see _nls_exposure_factor
        # docstring) -- only meaningful if the caller actually sent real sasa;
        # falls back to a neutral 1.0x (no-op) factor otherwise, same as the
        # RSA=0.4 neutral default used everywhere else in this file.
        mean_rsa = float(np.mean(sasa)) if sasa else 0.4
        exposure_factor = _nls_exposure_factor(mean_rsa) if sasa else 1.0
        raw_probability = result['nls_probability']
        gated_probability = raw_probability * exposure_factor

        return jsonify({
            'sequence': sequence,
            'nls_probability': round(gated_probability, 3),
            'raw_nls_probability': round(raw_probability, 3),
            'accessibility_rsa': round(mean_rsa, 3) if sasa else None,
            'exposure_factor': round(exposure_factor, 3),
            'predicted_class': result['predicted_class'],
            'pssm_score': round(result['pssm_score'], 3),
            'feature_importance': feature_importance,
        })

    except Exception as e:
        print(f"Error in NLS prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _nls_exposure_factor(rsa):
    """NLS accessibility gate -- the NLS-side analog of the NES exposure_factor
    in calculate_improved_nes_score(). Added after re-running
    diagnose_feature_importance_nls.py on the corrected, Tien et al. 2013
    residue-normalized RSA feature (see nls_data_pipeline/structural_data.json,
    regenerated after the flat-divisor SASA bug fix):

      sasa_norm: univariate ROC-AUC = 0.734, point-biserial r = +0.377
      (p = 2.1e-26) against the real positive/negative NLS dataset -- the
      5th/6th strongest of 19 features by both model-free checks. That is a
      real, statistically robust signal that importin-binding NLS motifs skew
      toward higher accessibility, consistent with the same physical logic
      already applied on the NES/CRM1 side: a motif buried in a folded
      domain core cannot be read by the import machinery, however good its
      sequence pattern looks.

      Notably, the trained random forest's own permutation importance for
      sasa_norm is ~0 (it isn't leaning on the feature much, likely because
      it's partly redundant with frac_basic/pssm_score once those are already
      in the model) -- impurity importance still ranks it in the top third of
      features, though. Since this is a physical constraint rather than
      something we want to leave to a possibly-redundant learned coefficient,
      it's applied here explicitly, same as the NES side, but with a gentler
      slope: classical NLS-bearing regions (unlike the specific CRM1
      hydrophobic groove interaction) tolerate more moderate burial, and the
      univariate signal here (AUC 0.73), while real, is weaker than the NES
      side's dominant feature (flank_hpr_likelihood, AUC 0.90).
    """
    if rsa >= 0.30:
        return 1.0
    elif rsa >= 0.10:
        return 0.6 + ((rsa - 0.10) / 0.20) * 0.4
    else:
        return 0.3 + (rsa / 0.10) * 0.3


@app.route('/api/nls_scan', methods=['POST', 'OPTIONS'])
def nls_scan():
    """
    WHOLE-STRUCTURE NLS SCAN -- the structural analog of /api/nls_predict,
    mirroring how /api/unified_crm1_nes scans a full sequence for NES
    motifs rather than scoring one pasted candidate.

    Unlike the NES scan, this does NOT re-download/re-parse the PDB or run
    fpocket -- there's no single dominant cargo receptor pocket to dock
    against for classical NLS import the way CRM1 anchors the NES side
    (see NLS_predictor_landscape_and_novelty.md), so this route is much
    lighter: it takes the sequence + per-residue pLDDT/SASA the frontend
    already has loaded in structureData (same pattern as /api/analyze_region)
    and slides candidate windows across it with NLSPredictor.scan_sequence(),
    which pre-filters with the monopartite/bipartite consensus regex before
    running the trained classifier, then greedily removes overlaps by score.
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        if nls_predictor is None or nls_predictor.model is None:
            return jsonify({'error': 'NLS predictor not available -- run '
                             '`python nls_ml_predictor.py train` in the AlphaFold '
                             'directory to train and save a model first.'}), 503

        data = request.json or {}
        sequence = data.get('sequence', '')
        plddt_values = data.get('plddt', None)
        sasa_values = data.get('sasa', None)
        # Whole-protein consensus_z/agreement_sd, now sent up
        # front by /api/structure (see get_structure()) -- lets this route
        # build a real rsa_profile per NLS candidate (same shape as the NES
        # side's) purely by slicing arrays it's already given, no second
        # structure download needed (keeping this endpoint "light", per the
        # docstring above).
        consensus_z_values = data.get('consensus_z', None)
        agreement_sd_values = data.get('agreement_sd', None)
        model_id = data.get('model_id', 'unknown')
        # Same explicit-uniprot_id-over-parsed-model_id
        # precedent as unified_crm1_nes_analysis (see its comment) -- lets
        # this route fetch real UniProt lipidation/subcellular-location
        # evidence (see fetch_uniprot_lipidation_annotations() docstring)
        # without assuming model_id is always "AF-{accession}-F1"-shaped.
        uniprot_id = data.get('uniprot_id')
        if not uniprot_id and isinstance(model_id, str) and model_id.startswith('AF-'):
            parts = model_id.split('-')
            if len(parts) >= 2:
                uniprot_id = parts[1]
        elif not uniprot_id and isinstance(model_id, str) and model_id not in ('', 'unknown'):
            # (bugfix): confirmed directly against a real holdout
            # run -- run_nls_holdout_pipeline_test.py sends model_id as a
            # bare UniProt accession (e.g. "P29966"), not the
            # "AF-{accession}-F1" AlphaFold entryId shape the branch above
            # assumes. That silently left uniprot_id=None for all 50
            # candidates, so fetch_uniprot_lipidation_annotations() never
            # ran and MARCKS/GAP-43/LL-37 stayed false positives even after
            # the veto was added. Any caller that already knows the plain
            # accession (not just ones using the AF- entryId convention)
            # should still get real UniProt evidence looked up.
            uniprot_id = model_id

        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400

        print(f"\n{'='*80}")
        print(f"NLS STRUCTURE SCAN: {model_id} ({len(sequence)} residues)")
        print(f"{'='*80}")
        start_time = time.time()

        regions = nls_predictor.scan_sequence(sequence, plddt_values=plddt_values,
                                               sasa_values=sasa_values)

        # Real UniProt lipidation/subcellular-location evidence (see
        # fetch_uniprot_lipidation_annotations() docstring) -- fetched once
        # per protein, applied to every surviving candidate below, same
        # "protein-level real annotation" pattern as the NES side's
        # coiled-coil check, just wired all the way into scoring here
        # (rather than log-only) since it was already validated against
        # 3 real holdout false positives before being added.
        lipid_sites, comment_texts, location_text = [], [], ''
        mature_chain_ranges, cleaved_ranges = [], []
        dna_binding_regions = []
        if uniprot_id:
            (lipid_sites, comment_texts, location_text,
             mature_chain_ranges, cleaved_ranges) = fetch_uniprot_lipidation_annotations(uniprot_id)
            dna_binding_regions = fetch_uniprot_dna_binding_regions(uniprot_id)

        # Apply the accessibility gate (see _nls_exposure_factor docstring)
        # using the real per-residue RSA this endpoint already has, same
        # pattern as the NES side's exposure_factor discount. Keeps the raw
        # ML probability visible too, so the effect of the gate is auditable
        # rather than silently baked in.
        for r in regions:
            r['raw_nls_probability'] = r['nls_probability']
            if sasa_values is not None and len(sasa_values) >= r['end'] + 1:
                region_rsa = sasa_values[r['start']:r['end'] + 1]
                mean_rsa = float(np.mean(region_rsa)) if region_rsa else 0.4
            else:
                mean_rsa = 0.4  # unknown -- neutral, matches calculate_sasa()'s own fallback
            r['accessibility_rsa'] = round(mean_rsa, 3)
            r['exposure_factor'] = round(_nls_exposure_factor(mean_rsa), 3)
            r['nls_probability'] = r['raw_nls_probability'] * r['exposure_factor']

            # Membrane-anchor/secreted veto (see fetch_uniprot_lipidation_
            # annotations() / nonnuclear_anchor_factor() docstrings) --
            # per-candidate-window v4), not purely protein-level
            # like earlier versions: a covalent lipid site still applies to
            # the whole protein regardless of window (unchanged), but
            # location-text evidence is now also checked against whether
            # THIS window survives into an annotated mature UniProt Chain,
            # so an isoform-specific 'Secreted' annotation can't wrongly
            # veto a window that only exists in a different, non-secreted
            # isoform (see nonnuclear_anchor_factor() for the full ORF2/
            # Lactoferrin case history).
            nonnuclear_anchored, anchor_factor, used_lipid_evidence = nonnuclear_anchor_factor(
                r['start'], r['end'], lipid_sites, comment_texts, mature_chain_ranges, cleaved_ranges)
            r['uniprot_nonnuclear_anchored'] = nonnuclear_anchored
            # Unconditional debug field (r['uniprot_location'] below
            # is only ever set when the veto actually fires) -- added while
            # investigating why CXCL12 (P48061) stayed a false positive with
            # this veto never engaging. Without this, there was no way to tell
            # from the API response alone whether comment_texts came back
            # empty/fetch-failed, or came back real but just didn't contain a
            # trigger keyword, or contained one but got excluded by the
            # mature-chain check -- all three look identical from outside
            # (uniprot_nonnuclear_anchored=False) without seeing the raw text.
            r['uniprot_location_raw_debug'] = location_text
            if nonnuclear_anchored:
                r['raw_nls_probability_pre_anchor_veto'] = r['nls_probability']
                r['nls_probability'] = r['nls_probability'] * anchor_factor
                r['uniprot_location'] = location_text
                r['anchor_factor'] = anchor_factor
                strength = ("a real covalent lipidation site" if used_lipid_evidence
                            else "curated location text alone (no lipidation site)")
                r['anchor_caveat'] = (
                    f"UniProt curates this protein's location as '{location_text}', with real "
                    f"lipidation evidence at {[s['position'] for s in lipid_sites]}" if used_lipid_evidence else
                    f"UniProt curates this protein's location as '{location_text}'"
                ) + (
                    f" and no nuclear localization annotation -- nls_probability has been discounted "
                    f"by a factor of {anchor_factor} (based on {strength}), since this reads like a "
                    f"membrane-anchored or secreted effector, not nuclear import cargo."
                )

            # DNA-binding-domain veto (see fetch_uniprot_dna_binding_regions()/
            # dna_binding_domain_factor() docstrings) -- the hardest documented
            # NLS failure mode, per-candidate (not protein-level like the
            # anchor veto above) since a protein can have a real NLS elsewhere
            # and a separate real DNA-binding domain that merely LOOKS basic.
            db_factor = dna_binding_domain_factor(r['start'], r['end'], dna_binding_regions)
            r['dna_binding_domain_factor'] = db_factor if db_factor < 1.0 else None
            if db_factor < 1.0:
                r['raw_nls_probability_pre_dna_binding_veto'] = r['nls_probability']
                r['nls_probability'] = r['nls_probability'] * db_factor
                r['dna_binding_caveat'] = (
                    f"This window substantially overlaps a real UniProt-curated 'DNA binding' "
                    f"region (not a learned feature -- see this project's own dna_binding_hard "
                    f"training negatives). nls_probability has been discounted by a factor of "
                    f"{round(db_factor, 2)}, since a basic patch inside a real DNA-binding domain "
                    f"reads as ordinary DNA-contact chemistry rather than a nuclear import signal."
                )

        # Re-sort/re-filter now that the gate may have moved scores, same
        # 0.5 cutoff scan_sequence() itself uses -- a region that only
        # passed on unadjusted probability but is clearly buried shouldn't
        # survive the gate silently disagreeing with the displayed score.
        # Added the same py_nls_shaped exception scan_sequence()
        # already has (nls_ml_predictor.py, -- without it, this
        # second filter was silently re-dropping PY-NLS candidates (e.g.
        # hnRNP A1's M9 domain) that scan_sequence() deliberately let
        # through regardless of score, since nls_probability isn't
        # meaningful for that class (see py_nls_caveat). The accessibility
        # gate above still runs and still adjusts nls_probability for
        # display/sorting either way; only the survival cutoff is bypassed.
        regions = [r for r in regions if r['nls_probability'] > 0.5 or r.get('py_nls_shaped')]
        regions.sort(key=lambda r: r['nls_probability'], reverse=True)

        # CIDER + RSA profiles for the surviving candidates only (computed
        # after the gate/filter above, not before, so rejected windows don't
        # waste the work) -- same +/-20 residue flanking window and same
        # profile shape as the NES side's cider_profile/rsa_profile (see
        # unified_crm1_nes_analysis), just built from arrays this endpoint
        # already has rather than a fresh structure download. 'positions'
        # here is simple 1-based sequential numbering into `sequence`
        # (this route has no PDB-derived residue_numbers with potential
        # gaps the way the NES route does -- nls_binding_regions' own
        # start/end below use the same convention).
        cider_rsa_flank = 20
        for r in regions:
            ctx_start = max(0, r['start'] - cider_rsa_flank)
            ctx_end = min(len(sequence), r['end'] + 1 + cider_rsa_flank)
            ctx_seq = sequence[ctx_start:ctx_end]
            cider_raw = compute_linear_cider_profiles(ctx_seq)
            r['cider_profile'] = {
                'positions': list(range(ctx_start + 1, ctx_end + 1)),
                'linear_hydropathy': cider_raw['linear_hydropathy'],
                'linear_ncpr': cider_raw['linear_ncpr'],
                'linear_fcr': cider_raw['linear_fcr'],
                'linear_complexity': cider_raw['linear_complexity'],
                'cider_computed': cider_raw['cider_computed'],
                'nes_start_idx_in_profile': r['start'] - ctx_start,
                'nes_end_idx_in_profile': r['end'] - ctx_start,
            }

            rsa_computed = sasa_values is not None and len(sasa_values) >= ctx_end
            r['rsa_profile'] = {
                'positions': list(range(ctx_start + 1, ctx_end + 1)),
                'consensus_rsa': [round(v, 3) for v in sasa_values[ctx_start:ctx_end]] if rsa_computed else [],
                'consensus_z': (
                    [round(v, 3) if v is not None else None for v in consensus_z_values[ctx_start:ctx_end]]
                    if consensus_z_values is not None and len(consensus_z_values) >= ctx_end else []
                ),
                'agreement_sd': (
                    [round(v, 3) if v is not None else None for v in agreement_sd_values[ctx_start:ctx_end]]
                    if agreement_sd_values is not None and len(agreement_sd_values) >= ctx_end else []
                ),
                'rsa_computed': rsa_computed,
                'nes_start_idx_in_profile': r['start'] - ctx_start,
                'nes_end_idx_in_profile': r['end'] - ctx_start,
            }

        # Per-residue likelihood array for the 'nls' structure colour mode,
        # same shape/indexing convention as crm1_scores on the NES side.
        nls_scores = [0.0] * len(sequence)
        for r in regions:
            for i in range(r['start'], min(r['end'] + 1, len(nls_scores))):
                nls_scores[i] = max(nls_scores[i], r['nls_probability'])

        high_confidence = [r for r in regions if r['nls_probability'] > 0.75]
        medium_confidence = [r for r in regions if 0.6 < r['nls_probability'] <= 0.75]
        low_confidence = [r for r in regions if 0.5 < r['nls_probability'] <= 0.6]

        # 1-indexed start/end to match the rest of the app's residue-number
        # convention (crm1_binding_regions/region analysis are 1-indexed).
        nls_binding_regions = [
            {
                'sequence': r['sequence'],
                'start': r['start'] + 1,
                'end': r['end'] + 1,
                'length': r['length'],
                'nls_probability': round(r['nls_probability'], 3),
                'raw_nls_probability': round(r['raw_nls_probability'], 3),
                'accessibility_rsa': r['accessibility_rsa'],
                'exposure_factor': r['exposure_factor'],
                'predicted_class': r['predicted_class'],
                'is_bipartite': r['is_bipartite'],
                'pssm_score': round(r['pssm_score'], 3),
                'caax_membrane_anchor': r.get('caax_membrane_anchor', False),
                'caax_caveat': r.get('caax_caveat'),
                'basic_background_veto': r.get('basic_background_veto', False),
                'basic_background_caveat': r.get('basic_background_caveat'),
                'basic_background_factor': r.get('basic_background_factor'),
                'uniprot_nonnuclear_anchored': r.get('uniprot_nonnuclear_anchored', False),
                'uniprot_location': r.get('uniprot_location'),
                'anchor_caveat': r.get('anchor_caveat'),
                'anchor_factor': r.get('anchor_factor'),
                'dna_binding_domain_factor': r.get('dna_binding_domain_factor'),
                'dna_binding_caveat': r.get('dna_binding_caveat'),
                'potential_tripartite': r.get('potential_tripartite', False),
                'tripartite_note': r.get('tripartite_note'),
                'tripartite_extra_cluster': r.get('tripartite_extra_cluster'),
                'cider_profile': r['cider_profile'],
                'rsa_profile': r['rsa_profile'],
            }
            for r in regions
        ]

        elapsed = time.time() - start_time
        print(f"   Found {len(regions)} candidate NLS regions "
              f"({len(high_confidence)} high / {len(medium_confidence)} medium / "
              f"{len(low_confidence)} low confidence) in {elapsed:.1f}s")
        print(f"{'='*80}\n")

        return jsonify({
            'nls_scores': nls_scores,
            'nls_binding_regions': nls_binding_regions,
            'summary': {
                'filtered_predictions': len(nls_binding_regions),
                'high_confidence': len(high_confidence),
                'medium_confidence': len(medium_confidence),
                'low_confidence': len(low_confidence),
                'analysis_time': round(elapsed, 2),
            }
        })

    except Exception as e:
        print(f"Error in NLS scan: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhanced_crm1/<model_id>', methods=['GET'])
def enhanced_crm1_analysis(model_id):
    """
    NEW: Enhanced CRM1 analysis combining all methods
    """
    try:
        print(f"\n{'='*60}")
        print(f"ENHANCED CRM1 ANALYSIS: {model_id}")
        print(f"{'='*60}")

        # Download structure
        pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
        response = requests.get(pdb_url, timeout=30)

        if response.status_code != 200:
            return jsonify({'error': 'Could not download structure'}), 404

        pdb_content = response.text

        # 1. Run pocket detection
        print("\n1. Detecting CRM1-binding pockets...")
        pockets = pocket_detector.detect_pockets(pdb_content)

        # 2. Get structure data (reuse existing code)
        # Parse structure and extract sequence
        from Bio.PDB import PDBParser
        from io import StringIO

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', StringIO(pdb_content))

        residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':
                        residues.append(residue)

        # Extract sequence
        try:
            from Bio.PDB.Polypeptide import three_to_one
        except ImportError:
            def three_to_one(residue_name):
                three_to_one_dict = {
                    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                }
                return three_to_one_dict.get(residue_name.upper(), 'X')

        sequence = ''.join([three_to_one(r.get_resname()) for r in residues])

        print(f"   Sequence length: {len(sequence)}")

        # 3. Scan for NES motifs with ML
        print("\n2. Scanning for NES with ML predictions...")
        window_size = 10
        ml_enhanced_motifs = []

        for i in range(len(sequence) - window_size + 1):
            subseq = sequence[i:i+window_size]

            # ML prediction
            result = ml_predictor.predict(subseq)
            # Handle both old (2 values) and new (3 values) return format
            if len(result) == 3:
                ml_prob, ml_conf, ml_details = result
            else:
                ml_prob, ml_conf = result

            # Only report high-probability motifs
            if ml_prob > 0.6:
                # Calculate average hydrophobicity for display (not scoring)
                hydro_values = [HYDROPHOBICITY.get(aa, 0.0) for aa in subseq]
                avg_hydrophobicity = float(np.mean(hydro_values)) if hydro_values else 0.0

                ml_enhanced_motifs.append({
                    'start': i + 1,
                    'end': i + window_size,
                    'sequence': subseq,
                    'full_sequence': sequence,  # Add full sequence for extended helix analysis
                    'ml_probability': round(ml_prob, 3),
                    'ml_confidence': ml_conf,
                    'avg_hydrophobicity': round(avg_hydrophobicity, 3),  # Display only, not in score
                    'method': 'machine_learning'
                })

        print(f"   Found {len(ml_enhanced_motifs)} high-probability ML motifs")

        # 3. Return combined results
        return jsonify({
            'model_id': model_id,
            'sequence_length': len(sequence),
            'pockets': pockets,
            'ml_nes_predictions': ml_enhanced_motifs,
            'total_pockets': len(pockets),
            'total_ml_motifs': len(ml_enhanced_motifs)
        })

    except Exception as e:
        print(f"Error in enhanced CRM1 analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MD ENDPOINTS FOR CRM1 BINDING ANALYSIS
# ============================================================================

@app.route('/api/md_docking', methods=['POST', 'OPTIONS'])
def md_docking():
    """
    Main MD Docking endpoint - Frontend-compatible
    Runs MD simulations on the submitted NES candidate(s) with CRM1 docking
    (up to MAX_MD_CANDIDATES). Frontend controls whether that's 1 candidate
    or the top 10 via the "candidates" list it sends.
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200

    try:
        if not MD_AVAILABLE:
            return jsonify({
                'error': 'MD simulations not available',
                'message': 'OpenMM and MD modules are required. Install with: conda install -c conda-forge openmm pdbfixer'
            }), 503

        data = request.json
        model_id = data.get('model_id')
        candidates = data.get('candidates', [])
        duration_ns = data.get('duration_ns', 10.0)  # Default 10 ns

        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400

        if not candidates:
            return jsonify({'error': 'No candidates provided'}), 400

        # How many candidates to actually simulate. The frontend already
        # sends exactly the candidates it wants (1 for "single candidate"
        # mode, 10 for "top 10" mode) -- MAX_MD_CANDIDATES is just a safety
        # ceiling against a caller sending an unreasonably large list, not
        # the thing driving the 1-vs-10 choice.
        MAX_MD_CANDIDATES = 10
        top_candidates = sorted(
            candidates,
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )[:MAX_MD_CANDIDATES]

        print(f"\n{'='*60}")
        print(f"MD DOCKING: {model_id}")
        print(f"{'='*60}")
        print(f"  {len(top_candidates)} candidate(s) selected for MD docking")
        print(f"  Simulation duration: {duration_ns} ns per candidate")

        # Download PDB structure
        pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
        response = requests.get(pdb_url, timeout=30)

        if response.status_code != 200:
            return jsonify({'error': 'Could not download structure'}), 404

        pdb_content = response.text

        # Submit job to queue
        job_id = job_queue.submit_job(
            model_id=model_id,
            pdb_content=pdb_content,
            nes_candidates=top_candidates,
            duration_ns=duration_ns
        )

        print(f"MD job submitted: {job_id}")

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'num_candidates': len(top_candidates),
            'duration_ns': duration_ns,
            'mode': 'docking',
            'message': f'MD simulation queued for {len(top_candidates)} NES candidates'
        })

    except Exception as e:
        print(f"Error submitting MD docking job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/crm1_md_docking/<model_id>', methods=['POST'])
def crm1_md_docking(model_id):
    """
    LEGACY: Advanced CRM1 Binding Molecular Dynamics
    Runs MD simulations on top 5 NES candidates with CRM1 docking
    (Kept for backward compatibility)
    """
    try:
        if not MD_AVAILABLE:
            return jsonify({'error': 'MD simulations not available',
                             'message': 'OpenMM and MD modules are required. Install with: conda install -c conda-forge openmm pdbfixer'}), 503

        data = request.json
        nes_candidates = data.get('nes_candidates', [])
        duration_ns = data.get('duration_ns', 10.0)

        if not nes_candidates:
            return jsonify({'error': 'No NES candidates provided'}), 400

        MAX_MD_CANDIDATES = 10

        top_candidates = sorted(
            nes_candidates,
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )[:MAX_MD_CANDIDATES]

        print(f"\n{'='*60}")
        print(f"CRM1 MD DOCKING: {model_id}")
        print(f"{'='*60}")
        print(f"  {len(top_candidates)} NES candidate(s) selected for MD docking")
        print(f"  Simulation duration: {duration_ns} ns per candidate")

        pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
        response = requests.get(pdb_url, timeout=30)

        if response.status_code != 200:
            return jsonify({'error': 'Could not download structure'}), 404

        pdb_content = response.text

        job_id = job_queue.submit_job(
            model_id=model_id,
            pdb_content=pdb_content,
            nes_candidates=top_candidates,
            duration_ns=duration_ns
        )

        print(f"MD job submitted: {job_id}")

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'num_candidates': len(top_candidates),
            'duration_ns': duration_ns,
            'mode': 'docking',
            'message': f'MD simulation queued for {len(top_candidates)} NES candidates'
        })

    except Exception as e:
        print(f"Error submitting MD docking job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/md_job_status/<job_id>', methods=['GET', 'OPTIONS'])
def md_job_status(job_id):
    """
    Get status of an MD job
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        if not MD_AVAILABLE:
            return jsonify({'error': 'MD not available'}), 503

        status = job_queue.get_job_status(job_id)

        if status is None:
            return jsonify({'error': 'Job not found'}), 404

        return jsonify(status)

    except Exception as e:
        print(f"Error getting job status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/md_job_result/<job_id>', methods=['GET', 'OPTIONS'])
def md_job_result(job_id):
    """
    Get results of a completed MD job
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        if not MD_AVAILABLE:
            return jsonify({'error': 'MD not available'}), 503

        result = job_queue.get_job_result(job_id)

        return jsonify(result)

    except Exception as e:
        print(f"Error getting job result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/md_jobs', methods=['GET'])
def list_md_jobs():
    """
    List recent MD jobs
    """
    try:
        if not MD_AVAILABLE:
            return jsonify({'error': 'MD not available'}), 503

        limit = request.args.get('limit', 10, type=int)
        jobs = job_queue.list_jobs(limit=limit)

        return jsonify(jobs)

    except Exception as e:
        print(f"Error listing jobs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sumoylation/predict', methods=['POST'])
def predict_sumoylation():
    """Predict SUMOylation sites and analyze NES relationships"""
    try:
        data = request.json
        sequence = data.get('sequence', '')
        nes_candidates = data.get('nes_candidates', [])

        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400

        sumo_sites = sumo_predictor.predict_sumo_sites(sequence, min_score=0.4)

        nes_annotations = {}
        for candidate in nes_candidates:
            nes_start = candidate.get('start', 0)
            nes_end = candidate.get('end', 0)
            nes_seq = candidate.get('sequence', '')

            if nes_start and nes_end:
                analysis = sumo_predictor.analyze_sumo_nes_relationship(
                    sequence, nes_start, nes_end, sumo_sites
                )
                nes_key = f"{nes_start}_{nes_end}_{nes_seq}"
                nes_annotations[nes_key] = analysis

        return jsonify({
            'sumo_sites': sumo_sites,
            'total_sites': len(sumo_sites),
            'high_confidence_sites': len([s for s in sumo_sites if s['confidence'] == 'high']),
            'nes_annotations': nes_annotations
        })

    except Exception as e:
        print(f"SUMOylation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/quick-analysis', methods=['POST'])
def quick_analysis():
    """Quick helix/amphipathic analysis (fast alternative to MD)"""
    try:
        data = request.json
        candidates = data.get('candidates', [])

        if not candidates:
            return jsonify({'error': 'No candidates provided'}), 400

        enhanced_candidates = batch_quick_analysis(candidates)

        helix_scores = [c['quick_structural_analysis']['helix_score'] for c in enhanced_candidates]
        amphipathic_scores = [c['quick_structural_analysis']['amphipathic_score'] for c in enhanced_candidates]

        summary = {
            'total_analyzed': len(enhanced_candidates),
            'avg_helix_score': float(np.mean(helix_scores)) if helix_scores else 0,
            'avg_amphipathic_score': float(np.mean(amphipathic_scores)) if amphipathic_scores else 0,
            'favorable_count': sum(1 for c in enhanced_candidates if c['quick_structural_analysis']['is_favorable_for_nes']),
        }

        return jsonify({
            'candidates': enhanced_candidates,
            'summary': summary
        })

    except Exception as e:
        print(f"Quick analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/scoring-info', methods=['GET'])
def get_scoring_info():
    return jsonify({
        'description': 'Enhanced NES Scoring System',
        'components': [
            {'name': 'ML Prediction', 'weight': 0.25, 'description': 'Trained on 27 validated NES sequences using SVM'},
            {'name': 'CRM1 Pocket Binding', 'weight': 0.3, 'description': 'Hydrophobic groove compatibility around Cys528'},
            {'name': 'Hydrophobicity', 'weight': 0.15, 'description': 'Requires 4-5 hydrophobic anchors (L,I,V,F,M)'},
            {'name': 'Accessibility (RSA)', 'weight': 0.1, 'description': 'Consensus relative solvent accessibility (Tien et al. 2013 residue-normalized) for CRM1 accessibility'},
            {'name': 'Flexibility', 'weight': 0.1, 'description': 'Higher pLDDT variance indicates flexibility'},
            {'name': 'Disorder', 'weight': 0.1, 'description': 'AlphaFold-predicted and UniProt-annotated'},
        ]
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AlphaFold Protein Viewer - Backend Server")
    print("Using AlphaFold Database API v4")
    print("With support for multiple versions and fragments")
    print("With MD Refinement and CRM1 Docking")
    print("="*60)
    print("\nStarting Flask server...")
    print("Backend will be available at: http://localhost:5000")

    print("\nTesting AlphaFold API access...")

    # Quick test
    test_entries = get_alphafold_structure_info("P01308")  # Human insulin
    if test_entries:
        print(f"AlphaFold API is accessible!")
        print(f"  Found {len(test_entries)} entry/entries")
        for entry in test_entries:
            print(f"  Entry: {entry.get('entryId', 'N/A')}")
            print(f"  Versions: {entry.get('allVersions', [])}")
            if 'pdbUrl' in entry:
                print(f"  Latest PDB URL: {entry['pdbUrl']}")
    else:
        print(f"Warning: Could not access AlphaFold API")
        print(f"  Check your internet connection")
        print(f"  Or visit: https://alphafold.ebi.ac.uk/api/docs")

    print("="*60 + "\n")

    app.run(debug=True, port=5000)
