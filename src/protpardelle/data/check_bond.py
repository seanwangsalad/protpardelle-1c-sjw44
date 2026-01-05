#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import os

import warnings
import logging
warnings.filterwarnings('ignore')

# Suppress MDAnalysis info logs
logging.getLogger('MDAnalysis').setLevel(logging.ERROR)

# Van der Waals radii (in Angstroms)
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
}

# Bond connection cutoff tolerance
CONNECT_CUTOFF = 0.35  # Angstroms
SEARCH_RADIUS = 3.0  # Angstroms - search sphere radius

# Terminal atoms for each non-ring residue
TERMINAL_ATOMS = {
    'ALA': ['CB'],
    'ARG': ['NH1', 'NH2'],
    'ASN': ['OD1', 'ND2'],
    'ASP': ['OD1', 'OD2'],
    'CYS': ['SG'],
    'GLN': ['OE1', 'NE2'],
    'GLU': ['OE1', 'OE2'],
    'GLY': [],
    'ILE': ['CD1'],
    'LEU': ['CD1', 'CD2'],
    'LYS': ['NZ'],
    'MET': ['CE'],
    'SER': ['OG'],
    'THR': ['OG1', 'CG2'],
    'VAL': ['CG1', 'CG2'],
    'TYR': ['OH']
}

def get_vdw_radius(element):
    """Get VDW radius for an element, default to 1.7 if unknown"""
    return VDW_RADII.get(element, 1.7)

def check_bond(file):
    """Check if terminal atoms in non-ring residues are bonded to more than 1 atom"""
    u = mda.Universe(file)
    
    # Select non-ring residues
    non_ring_res = u.select_atoms("protein and not (resname TRP or resname PHE or resname HIS or resname PRO)").residues
    
    incorrect_bonds = 0
    
    for res in non_ring_res:
        res_name = res.resname
        
        if res_name not in TERMINAL_ATOMS:
            continue
        
        terminal_atoms = TERMINAL_ATOMS[res_name]
        
        for terminal_atom_name in terminal_atoms:
            # Select the terminal atom
            terminal_atom_sel = res.atoms.select_atoms(f"name {terminal_atom_name}")
            
            if len(terminal_atom_sel) == 0:
                continue
            
            terminal_atom = terminal_atom_sel[0]
            
            # Find atoms within search radius
            nearby = res.atoms.select_atoms(f"around {SEARCH_RADIUS} index {terminal_atom.index}")
            
            # Count bonds to this terminal atom
            bonded_count = 0
            
            for other_atom in nearby:
                if other_atom.index == terminal_atom.index:
                    continue
                
                # Check if bonded using distance criterion
                distance = np.linalg.norm(terminal_atom.position - other_atom.position)
                vdw1 = get_vdw_radius(terminal_atom.element)
                vdw2 = get_vdw_radius(other_atom.element)
                bond_cutoff = CONNECT_CUTOFF + (vdw1 + vdw2) / 2
                
                if distance <= bond_cutoff:
                    bonded_count += 1
            
            # If bonded to more than 1 atom, count as incorrect
            if bonded_count > 1:
                incorrect_bonds += 1
    
    return incorrect_bonds

if __name__ == "__main__":
    # Directory containing PDB files
    pdb_directory = "."  # Current directory, change as needed
    
    # Find all PDB files in directory
    pdb_files = [f for f in os.listdir(pdb_directory) if f.endswith('.pdb')]
    
    if len(pdb_files) == 0:
        print(f"No PDB files found in {pdb_directory}")
        exit()
    
    total_incorrect_bonds = 0
    
    # Process each PDB file
    for pdb_file in pdb_files:
        filepath = os.path.join(pdb_directory, pdb_file)
        try:
            incorrect = check_bond(filepath)
            total_incorrect_bonds += incorrect
            if incorrect > 0:
                print(f"{pdb_file}: {incorrect} incorrect bonds")
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Total incorrect bonds: {total_incorrect_bonds}")
    print(f"Structures analyzed: {len(pdb_files)}")
    print(f"Incorrect bonds per structure: {total_incorrect_bonds / len(pdb_files):.2f}")
    print(f"{'='*50}")