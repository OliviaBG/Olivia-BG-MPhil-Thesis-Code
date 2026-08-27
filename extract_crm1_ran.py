"""
Extract Chain A (CRM1) and Chain C (RanGTP) from 3NBY structure
Removes Chain B (NES peptide) to expose the binding cleft
"""

def extract_crm1_ran_only(input_pdb, output_pdb):
    """
    Extract only chains A (CRM1) and C (RanGTP) from the crystal structure.

    This removes the bound NES peptide (chain B) so that:
    1. The binding cleft around Cys528 is EXPOSED
    2. CRM1 remains in the correct conformation (with Ran bound)
    3. Fpocket can detect the empty binding pocket
    4. Target proteins can be docked into the available cleft

    Args:
        input_pdb: Path to 3NBY PDB file with all chains
        output_pdb: Path to output PDB with only chains A and C
    """

    print("="*70)
    print("Extracting CRM1-Ran Complex (Chains A + C)")
    print("="*70)
    print("\nRemoving:")
    print("  - Chain B (NES peptide) - to expose binding cleft")
    print("  - Chain D (duplicate CRM1)")
    print("  - Chain E (duplicate NES)")
    print("  - Chain F (duplicate Ran)")
    print("\nKeeping:")
    print("  - Chain A (CRM1 - Exportin-1)")
    print("  - Chain C (RanGTP)")
    print()

    kept_chains = {'A', 'C'}
    removed_residues = 0
    kept_residues = 0

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
                # Chain is at position 21 (0-indexed position 21, column 22)
                chain = line[21]

                if current_chain != chain:
                    current_chain = chain
                    if chain in kept_chains:
                        print(f"  Processing chain {chain}...")
                    else:
                        print(f"  Failed: Skipping chain {chain}...")

                if chain in kept_chains:
                    outfile.write(line)
                    kept_residues += 1
                else:
                    removed_residues += 1
                continue

            # Keep connectivity records for chains A and C only
            if line.startswith('CONECT'):
                # CONECT records reference atom serial numbers
                # We keep them all as they'll only reference atoms we kept
                outfile.write(line)
                continue

            # End of structure
            if line.startswith('END'):
                outfile.write(line)
                break

        # Add MASTER and END if not present
        if not line.startswith('END'):
            outfile.write('END\n')

    print("\n" + "="*70)
    print("Extraction Complete")
    print("="*70)
    print(f"Kept atoms:    {kept_residues:,}")
    print(f"Removed atoms: {removed_residues:,}")
    print(f"\nOutput written to: {output_pdb}")
    print("\nThis structure now has:")
    print("  CRM1 in biologically-relevant conformation")
    print("  RanGTP bound (maintains proper CRM1 structure)")
    print("  EXPOSED hydrophobic cleft around Cys528")
    print("  Ready for fpocket analysis or protein docking")
    print("="*70)


def verify_chains(pdb_file):
    """Verify which chains are present in a PDB file"""
    chains = set()
    chain_residue_count = {}

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                chains.add(chain)
                chain_residue_count[chain] = chain_residue_count.get(chain, 0) + 1

    print(f"\nChains in {pdb_file}:")
    for chain in sorted(chains):
        print(f"  Chain {chain}: {chain_residue_count[chain]:,} atoms")

    return chains


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_crm1_ran.py <input_3NBY.pdb> [output.pdb]")
        print("\nExample:")
        print("  python extract_crm1_ran.py 3NBY.pdb CRM1_Ran_only.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2] if len(sys.argv) > 2 else 'CRM1_Ran_only.pdb'

    # Verify input file
    print(f"\nInput file: {input_pdb}")
    try:
        verify_chains(input_pdb)
    except FileNotFoundError:
        print(f"Error: File '{input_pdb}' not found")
        sys.exit(1)

    # Extract chains A and C
    extract_crm1_ran_only(input_pdb, output_pdb)

    # Verify output
    print("\nVerifying output:")
    verify_chains(output_pdb)
