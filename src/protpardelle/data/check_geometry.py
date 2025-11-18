import MDAnalysis as mda
import numpy as np
import pandas as pd

import warnings
import logging
warnings.filterwarnings('ignore')

# Suppress MDAnalysis info logs
logging.getLogger('MDAnalysis').setLevel(logging.ERROR)

rings = {
    "TRP": ["CG", "CD2", "CE3", "CZ3", "CH2", "CZ2", "CE2", "NE1", "CD1"],
    "PHE": ["CG", "CD2", "CE2", "CZ", "CE1", "CD1"],
    "TYR": ["CG", "CD2", "CE2", "CZ", "CE1", "CD1"],
    "HIS": ["CG", "CD2", "NE2", "CE1", "ND1"],
    "PRO": ["CG", "CD", "N", "CA", "CB"]
}

def get_angle(A, B, C):
    AB = A - B
    CB = C - B
    cos_angle = np.dot(AB, CB) / (np.linalg.norm(AB) * np.linalg.norm(CB))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle) * 180 / np.pi 

def check_geometry(file):
    ref_stats = pd.read_csv('/home/seanwang/protpardelle-1c/src/protpardelle/data/ring_angles.csv')
    ref_dict = {}
    for _, row in ref_stats.iterrows():
        key = (row['residue_type'], row['atom_name'])
        ref_dict[key] = {
            'mean': row['mean_angle'],
            'std': row['std_angle']
        }

    # Load your PDB
    u = mda.Universe(file)
    ring_res = u.select_atoms("resname TRP or resname PHE or resname TYR or resname HIS or resname PRO").residues

    total_deviation = 0.0
    num_outliers = 0
    total_ring_residues = len(ring_res)

    for res in ring_res:
        res_name = res.resname
        res_id = res.resid
        
        try:
            sel_atoms = [res.atoms.select_atoms(f"name {i}").positions[0] for i in rings[res_name]]
            
            n = len(sel_atoms)
            for i in range(n):
                atom_name = rings[res_name][i]
                angle = get_angle(sel_atoms[(i - 1) % n], sel_atoms[i], sel_atoms[(i + 1) % n])
                
                # Get reference values
                key = (res_name, atom_name)
                if key in ref_dict:
                    mean_angle = ref_dict[key]['mean']
                    std_angle = ref_dict[key]['std']
                    
                    lower_bound = mean_angle - 1.5 * std_angle
                    upper_bound = mean_angle + 1.5 * std_angle
                    
                    # Check if outside bounds
                    if angle < lower_bound or angle > upper_bound:
                        deviation = abs(angle - mean_angle)
                        total_deviation += deviation
                        num_outliers += 1
                        
        except Exception as e:
            print(f"Error processing residue {res_name}{res_id}: {e}")
            continue

    # print(f"\nTotal ring residues: {total_ring_residues}")
    # print(f"Number of outlier angles: {num_outliers}")
    # print(f"Total absolute deviation: {total_deviation:.2f}°")
    # if num_outliers > 0:
    #     print(f"Average deviation per outlier: {total_deviation/num_outliers:.2f}°")
    # print(f"Total deviation per ring residue: {total_deviation/total_ring_residues:.2f}°")
    if total_ring_residues == 0:
        return 0
    else:
        return total_deviation, total_deviation / total_ring_residues
