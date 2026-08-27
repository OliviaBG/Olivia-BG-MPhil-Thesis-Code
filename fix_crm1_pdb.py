#!/usr/bin/env python3
"""
Script to fix CRM1 PDB file terminal issues
Run this ONCE to create a fixed version of your CRM1 PDB
"""

import sys
import os

def fix_crm1_pdb(input_path, output_path=None):
    """
    Fix CRM1 PDB file for OpenMM compatibility
    """
    try:
        from md_refinement import prepare_crm1_pdb_for_openmm
    except ImportError:
        print("Error: Cannot import md_refinement module")
        print("Make sure md_refinement.py is in the same directory")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if output_path is None:
        output_path = input_path.replace('.pdb', '_fixed.pdb')

    print("="*70)
    print("CRM1 PDB FIX UTILITY")
    print("="*70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("="*70)
    print()

    try:
        result_path = prepare_crm1_pdb_for_openmm(input_path, output_path)

        print()
        print("="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Fixed CRM1 PDB saved to: {result_path}")
        print()
        print("Next steps:")
        print("1. Update your app.py to use the fixed file:")
        print(f"   CRM1_REFERENCE_PATH = '{result_path}'")
        print()
        print("2. Restart your Flask server:")
        print("   python app.py")
        print("="*70)

        return result_path

    except Exception as e:
        print()
        print("="*70)
        print("ERROR FIXING PDB FILE")
        print("="*70)
        print(f"Error: {e}")
        print()
        print("This usually means:")
        print("1. OpenMM is not installed")
        print("   Install with: conda install -c conda-forge openmm")
        print()
        print("2. The PDB file has serious structural issues")
        print("   Try downloading the structure from RCSB PDB again")
        print()
        print("3. Missing force field files")
        print("   Reinstall OpenMM: conda install -c conda-forge openmm --force-reinstall")
        print("="*70)

        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_crm1_pdb.py <input.pdb> [output.pdb]")
        print()
        print("Example:")
        print("  python fix_crm1_pdb.py CRM1_Ran_only.pdb CRM1_Ran_fixed.pdb")
        print()
        print("If output is not specified, it will be input_fixed.pdb")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    fix_crm1_pdb(input_file, output_file)
