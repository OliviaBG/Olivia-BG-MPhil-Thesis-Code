"""
CRM1 Reference Structure Setup
Downloads 3NBY, extracts chains A+C, analyzes binding pocket
This creates the reference structure for pocket detection

Run this ONCE before starting the Flask app
"""

import requests
import numpy as np
from pathlib import Path
import json


def download_3nby(output_path):
    """Download 3NBY structure from RCSB PDB"""
    print("="*70)
    print("Step 1: Downloading 3NBY Crystal Structure")
    print("="*70)

    url = "https://files.rcsb.org/download/3NBY.pdb"

    print(f"\nDownloading from: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, 'w') as f:
            f.write(response.text)

        # Count lines
        lines = response.text.split('\n')
        atom_lines = [l for l in lines if l.startswith('ATOM')]

        print(f"Downloaded successfully")
        print(f"  File size: {len(response.text):,} bytes")
        print(f"  ATOM records: {len(atom_lines):,}")
        print(f"  Saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_chains_a_and_c(input_pdb, output_pdb):
    """
    Extract only chains A (CRM1) and C (RanGTP) from 3NBY
    Removes chain B (NES peptide) to expose the binding cleft
    """
    print("\n" + "="*70)
    print("Step 2: Extracting CRM1 + Ran (Chains A & C)")
    print("="*70)

    print("\nRemoving:")
    print("  - Chain B (NES peptide) - exposes binding cleft")
    print("  - Chain D (duplicate CRM1)")
    print("  - Chain E (duplicate NES)")
    print("  - Chain F (duplicate Ran)")
    print("\nKeeping:")
    print("  - Chain A (CRM1 - Exportin-1)")
    print("  - Chain C (RanGTP)")

    kept_chains = {'A', 'C'}
    kept_atoms = 0
    removed_atoms = 0

    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        current_chain = None

        for line in infile:
            # Keep header information
            if line.startswith(('HEADER', 'TITLE', 'COMPND', 'SOURCE',
                               'KEYWDS', 'EXPDTA', 'AUTHOR', 'REVDAT',
                               'REMARK', 'CRYST1')):
                outfile.write(line)
                continue

            # Process ATOM and HETATM records
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21]

                if current_chain != chain:
                    current_chain = chain
                    if chain in kept_chains:
                        print(f"\n  Processing chain {chain}...")
                    else:
                        print(f"  Failed: Skipping chain {chain}...")

                if chain in kept_chains:
                    outfile.write(line)
                    kept_atoms += 1
                else:
                    removed_atoms += 1
                continue

            # Keep connectivity records
            if line.startswith('CONECT'):
                outfile.write(line)
                continue

            # End of structure
            if line.startswith('END'):
                outfile.write(line)
                break

        # Ensure END is written
        if not line.startswith('END'):
            outfile.write('END\n')

    print("\n" + "="*70)
    print(f"Extraction Complete")
    print("="*70)
    print(f"Kept atoms:    {kept_atoms:,}")
    print(f"Removed atoms: {removed_atoms:,}")
    print(f"\nOutput: {output_pdb}")
    print("\nThis structure now has:")
    print("  CRM1 in biologically-relevant conformation")
    print("  RanGTP bound (maintains proper CRM1 structure)")
    print("  EXPOSED hydrophobic cleft around Cys528")
    print("  Ready for pocket detection")


def analyze_binding_pocket(pdb_path):
    """
    Analyze the CRM1 NES-binding pocket
    Returns pocket characteristics for detection
    """
    print("\n" + "="*70)
    print("Step 3: Analyzing CRM1 Binding Pocket")
    print("="*70)

    # Read PDB file
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    # Extract CRM1 atoms (chain A)
    crm1_residues_dict = {}

    print("\nParsing CRM1 structure (Chain A)...")

    for line in lines:
        if line.startswith('ATOM') and line[21] == 'A':
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_num = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            res_id = f"{res_name}{res_num}"

            if res_id not in crm1_residues_dict:
                crm1_residues_dict[res_id] = {
                    'atoms': [],
                    'ca_coord': None
                }

            crm1_residues_dict[res_id]['atoms'].append({
                'name': atom_name,
                'coords': np.array([x, y, z])
            })

            if atom_name == 'CA':
                crm1_residues_dict[res_id]['ca_coord'] = np.array([x, y, z])

    print(f"Found {len(crm1_residues_dict)} CRM1 residues")

    # Define the NES-binding region
    # Based on crystal structure analysis and literature
    binding_region_start = 360
    binding_region_end = 580

    print(f"\nAnalyzing NES-binding region: {binding_region_start}-{binding_region_end}")

    # Extract binding pocket residues (hydrophobic residues in binding region)
    hydrophobic = ['LEU', 'ILE', 'VAL', 'PHE', 'MET', 'TRP', 'ALA', 'CYS']
    binding_pocket = {}

    for res_id, res_data in crm1_residues_dict.items():
        # Extract residue number
        res_num = int(''.join(filter(str.isdigit, res_id)))
        res_name = ''.join(filter(str.isalpha, res_id))

        # Check if in binding region
        if binding_region_start <= res_num <= binding_region_end:
            # Include hydrophobic residues that form the groove
            if res_name in hydrophobic:
                binding_pocket[res_id] = res_data

    print(f"Identified {len(binding_pocket)} hydrophobic residues in binding pocket")

    # Calculate pocket geometry
    ca_coords = np.array([data['ca_coord'] for data in binding_pocket.values()
                          if data['ca_coord'] is not None])

    pocket_center = np.mean(ca_coords, axis=0)
    pocket_radius = np.max([np.linalg.norm(c - pocket_center) for c in ca_coords])

    print("\n" + "-"*70)
    print("POCKET GEOMETRY")
    print("-"*70)
    print(f"Center: ({pocket_center[0]:.2f}, {pocket_center[1]:.2f}, {pocket_center[2]:.2f})")
    print(f"Radius: {pocket_radius:.2f} Å")
    print(f"Span:   {pocket_radius * 2:.2f} Å")

    # Check for key residues
    key_residues = {
        'CYS528': None,
        'PHE366': None,
        'PHE414': None,
        'MET425': None,
        'LEU525': None
    }

    for key in key_residues.keys():
        if key in binding_pocket:
            key_residues[key] = binding_pocket[key]['ca_coord'].tolist()

    print("\n" + "-"*70)
    print("KEY RESIDUES")
    print("-"*70)

    for res, coord in key_residues.items():
        if coord:
            print(f"{res:10s} at ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f})")
        else:
            print(f"Failed: {res:10s} not found")

    # Critical check for Cys528
    if 'CYS528' in binding_pocket:
        print("\nCYS528 CONFIRMED - This is the critical cysteine in the binding cleft")
    else:
        print("\nWARNING: CYS528 not found in binding pocket!")

    # Return pocket characteristics
    pocket_data = {
        'center': pocket_center.tolist(),
        'radius': float(pocket_radius),
        'num_residues': len(binding_pocket),
        'residue_list': list(binding_pocket.keys()),
        'key_residues': key_residues,
        'has_cys528': 'CYS528' in binding_pocket,
        'binding_region': {
            'start': binding_region_start,
            'end': binding_region_end
        },
        'characteristics': {
            'type': 'elongated_hydrophobic_groove',
            'length_estimate': 25.0,
            'width_estimate': 10.0,
            'depth_estimate': 9.0
        }
    }

    return pocket_data


def setup_crm1_reference(data_dir='/data/crm1'):
    """
    Complete setup: Download, extract, and analyze CRM1 reference

    Args:
        data_dir: Directory to store reference files

    Returns:
        Path to CRM1_Ran_only.pdb reference file
    """
    print("\n" + "="*70)
    print(" "*15 + "CRM1 REFERENCE STRUCTURE SETUP")
    print(" "*20 + "PDB: 3NBY")
    print("="*70)

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"\nData directory: {data_path.absolute()}")

    # File paths
    pdb_3nby_original = data_path / '3NBY_original.pdb'
    crm1_ran_only = data_path / 'CRM1_Ran_only.pdb'
    pocket_analysis_json = data_path / 'pocket_analysis.json'

    # Step 1: Download 3NBY (if not already present)
    if pdb_3nby_original.exists():
        print(f"\n3NBY structure already exists: {pdb_3nby_original}")
    else:
        if not download_3nby(pdb_3nby_original):
            print("\nFailed: Setup failed: Could not download 3NBY")
            return None

    # Step 2: Extract chains A + C
    print("\n")
    extract_chains_a_and_c(pdb_3nby_original, crm1_ran_only)

    # Step 3: Analyze binding pocket
    print("\n")
    pocket_data = analyze_binding_pocket(crm1_ran_only)

    # Step 4: Save analysis
    print("\n" + "="*70)
    print("Step 4: Saving Analysis")
    print("="*70)

    with open(pocket_analysis_json, 'w') as f:
        json.dump(pocket_data, f, indent=2)

    print(f"Pocket analysis saved to: {pocket_analysis_json}")

    # Summary
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print(f"\nReference files created:")
    print(f"  1. Original structure:  {pdb_3nby_original}")
    print(f"  2. CRM1+Ran reference:  {crm1_ran_only}")
    print(f"  3. Pocket analysis:     {pocket_analysis_json}")

    print(f"\nUse this reference path in app.py:")
    print(f"  CRM1_REFERENCE_PATH = '{crm1_ran_only.absolute()}'")

    print("\n" + "="*70)

    return str(crm1_ran_only.absolute())


if __name__ == '__main__':
    import sys

    # Allow custom data directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/data/crm1'

    # If /data is not writable, use current directory
    if not Path('/data').exists():
        data_dir = './crm1_reference'
        print(f"Warning: /data directory not found, using: {data_dir}")

    try:
        reference_path = setup_crm1_reference(data_dir)

        if reference_path:
            print("\nSUCCESS! CRM1 reference is ready.")
            print(f"\nNext steps:")
            print(f"  1. Update app.py to use: {reference_path}")
            print(f"  2. Run: python app.py")
            print(f"  3. Start analyzing proteins!")
        else:
            print("\nFailed: Setup failed. Check errors above.")
            sys.exit(1)

    except Exception as e:
        print(f"\nFailed: Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
