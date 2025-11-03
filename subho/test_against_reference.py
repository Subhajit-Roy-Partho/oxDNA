#!/usr/bin/env python3
"""
Test script for oxDNA2 PyTorch implementation against reference energies.

This script:
1. Reads topology from example/dsdna8.top
2. Reads configuration from example/init.dat
3. Computes energy components using oxdna2_pytorch.py
4. Compares results with reference JSON files (output_*.json)
5. Reports errors for each nucleotide and overall
"""

import torch
import numpy as np
import json
import sys
import os
from pathlib import Path

# Import the oxDNA2 energy module
from oxdna2_pytorch import oxDNA2Energy


def read_topology(topology_file):
    """
    Read oxDNA topology file.

    Returns:
        n_nucleotides: number of nucleotides
        n_strands: number of strands
        base_types: list of base type indices (0=A, 1=C, 2=G, 3=T)
        strand_ids: list of strand IDs for each nucleotide
        n3_neighbors: list of 3' neighbor indices (-1 if none)
        n5_neighbors: list of 5' neighbor indices (-1 if none)
    """
    base_to_number = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    with open(topology_file, 'r') as f:
        first_line = f.readline().split()
        n_nucleotides = int(first_line[0])
        n_strands = int(first_line[1])

        base_types = []
        strand_ids = []
        n3_neighbors = []
        n5_neighbors = []

        for line in f:
            parts = line.split()
            strand_id = int(parts[0])
            base = parts[1]
            n3 = int(parts[2])
            n5 = int(parts[3])

            strand_ids.append(strand_id)
            base_types.append(base_to_number[base])
            n3_neighbors.append(n3)
            n5_neighbors.append(n5)

    return n_nucleotides, n_strands, base_types, strand_ids, n3_neighbors, n5_neighbors


def read_configuration(config_file, n_nucleotides):
    """
    Read oxDNA configuration file.

    Returns:
        time: simulation time
        box: box dimensions (3,)
        energy: [E_tot, E_pot, E_kin]
        positions: nucleotide center-of-mass positions (n_nucleotides, 3)
        orientations: nucleotide orientation matrices (n_nucleotides, 3, 3)
        velocities: nucleotide velocities (n_nucleotides, 3)
        angular_velocities: nucleotide angular velocities (n_nucleotides, 3)
    """
    with open(config_file, 'r') as f:
        # Read header
        time_line = f.readline().split()
        time = float(time_line[2])

        box_line = f.readline().split()
        box = np.array([float(x) for x in box_line[2:5]])

        energy_line = f.readline().split()
        energy = [float(x) for x in energy_line[2:5]]

        # Read nucleotide data
        positions = []
        orientations = []
        velocities = []
        angular_velocities = []

        for i in range(n_nucleotides):
            line = f.readline().split()
            cm = np.array([float(x) for x in line[0:3]])
            a1 = np.array([float(x) for x in line[3:6]])
            a3 = np.array([float(x) for x in line[6:9]])
            v = np.array([float(x) for x in line[9:12]])
            L = np.array([float(x) for x in line[12:15]])

            # Construct a2 as a3 x a1 (right-handed coordinate system)
            a2 = np.cross(a3, a1)

            # Build orientation matrix (columns are a1, a2, a3)
            orientation = np.column_stack([a1, a2, a3])

            positions.append(cm)
            orientations.append(orientation)
            velocities.append(v)
            angular_velocities.append(L)

    positions = np.array(positions)
    orientations = np.array(orientations)
    velocities = np.array(velocities)
    angular_velocities = np.array(angular_velocities)

    return time, box, energy, positions, orientations, velocities, angular_velocities


def read_reference_energies(json_file):
    """Read reference energies from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Get the first (and only) key-value pair
    key = list(data.keys())[0]
    energies = np.array(data[key])
    return key, energies


def compute_all_pairwise_energies(model, positions, orientations, base_types,
                                  n3_neighbors, n5_neighbors):
    """
    Compute all energy components for each nucleotide.

    Energy is accumulated per nucleotide (counting each pairwise interaction once,
    attributed to the lower-indexed nucleotide).
    """
    n = len(positions)

    # Initialize energy arrays per nucleotide
    energies_per_nuc = {
        'FENE': np.zeros(n),
        'BEXC': np.zeros(n),  # Bonded excluded volume
        'STCK': np.zeros(n),  # Stacking
        'NEXC': np.zeros(n),  # Non-bonded excluded volume
        'HB': np.zeros(n),    # Hydrogen bonding
        'CRSTCK': np.zeros(n),  # Cross-stacking
        'CXSTCK': np.zeros(n),  # Coaxial stacking
        'DH': np.zeros(n),    # Debye-Huckel (not in reference files, but computed)
        'total': np.zeros(n)
    }

    # Convert to tensors
    pos_t = torch.tensor(positions, dtype=torch.float32)
    orient_t = torch.tensor(orientations, dtype=torch.float32)
    base_t = torch.tensor(base_types, dtype=torch.long)

    # Identify terminus nucleotides
    is_terminus = [(n3 == -1 or n5 == -1) for n3, n5 in zip(n3_neighbors, n5_neighbors)]

    # Process all pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Check if bonded (i is 5' neighbor of j or vice versa)
            is_bonded_pair = (n3_neighbors[i] == j) or (n5_neighbors[i] == j)

            # Prepare tensors for this pair
            pos_p = pos_t[i:i+1]
            pos_q = pos_t[j:j+1]
            orient_p = orient_t[i:i+1]
            orient_q = orient_t[j:j+1]
            base_p = base_t[i:i+1]
            base_q = base_t[j:j+1]
            is_bonded_t = torch.tensor([is_bonded_pair])
            is_term_p = torch.tensor([is_terminus[i]])
            is_term_q = torch.tensor([is_terminus[j]])

            if is_bonded_pair:
                # Ensure p is the 5' neighbor of q for stacking
                p_idx, q_idx = i, j
                if n3_neighbors[i] != j:
                    p_idx, q_idx = j, i

                # Prepare tensors for this pair in the correct 5'->3' order
                pos_p_st = pos_t[p_idx:p_idx+1]
                pos_q_st = pos_t[q_idx:q_idx+1]
                orient_p_st = orient_t[p_idx:p_idx+1]
                orient_q_st = orient_t[q_idx:q_idx+1]
                base_p_st = base_t[p_idx:p_idx+1]
                base_q_st = base_t[q_idx:q_idx+1]
                
                # Compute bonded energies
                # FENE
                a1_p = orient_p[0, :, 0]
                a1_q = orient_q[0, :, 0]
                back_p = pos_p[0] + a1_p * (-0.4)  # POS_BACK
                back_q = pos_q[0] + a1_q * (-0.4)
                r_backbone = torch.norm(back_q - back_p)
                fene_energy = model.backbone_energy(r_backbone.unsqueeze(0)).item()
                energies_per_nuc['FENE'][i] += fene_energy

                # Bonded excluded volume
                bexc_energy = model.bonded_excluded_volume(pos_p, pos_q, orient_p, orient_q).item()
                energies_per_nuc['BEXC'][i] += bexc_energy

                # Stacking
                stck_energy = model.stacking_energy(pos_p_st, pos_q_st, orient_p_st, orient_q_st,
                                                   base_p_st, base_q_st).item()
                energies_per_nuc['STCK'][i] += stck_energy

            else:
                # Compute non-bonded energies
                # Non-bonded excluded volume
                nexc_energy = model.nonbonded_excluded_volume(pos_p, pos_q, orient_p, orient_q).item()
                energies_per_nuc['NEXC'][i] += nexc_energy

                # Hydrogen bonding
                hb_energy = model.hydrogen_bonding_energy(pos_p, pos_q, orient_p, orient_q,
                                                         base_p, base_q).item()
                energies_per_nuc['HB'][i] += hb_energy

                # Cross-stacking
                crstck_energy = model.cross_stacking_energy(pos_p, pos_q, orient_p, orient_q).item()
                energies_per_nuc['CRSTCK'][i] += crstck_energy

                # Coaxial stacking
                cxstck_energy = model.coaxial_stacking_energy(pos_p, pos_q, orient_p, orient_q).item()
                energies_per_nuc['CXSTCK'][i] += cxstck_energy

            # Debye-Huckel (all non-bonded pairs)
            dh_energy = model.debye_huckel_energy(pos_p, pos_q, orient_p, orient_q,
                                                 is_bonded_t, is_term_p, is_term_q).item()
            energies_per_nuc['DH'][i] += dh_energy

    # Compute total
    for key in ['FENE', 'BEXC', 'STCK', 'NEXC', 'HB', 'CRSTCK', 'CXSTCK', 'DH']:
        energies_per_nuc['total'] += energies_per_nuc[key]

    return energies_per_nuc


def compare_energies(computed, reference, component_name):
    """
    Compare computed energies with reference and report errors.

    Returns:
        per_nucleotide_errors: array of absolute errors per nucleotide
        rmse: root mean square error
        max_error: maximum absolute error
        mean_error: mean absolute error
    """
    errors = np.abs(computed - reference)
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    mean_error = np.mean(errors)

    return errors, rmse, max_error, mean_error


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    example_dir = script_dir / "example"

    topology_file = example_dir / "dsdna8.top"
    config_file = example_dir / "init.dat"

    print("=" * 80)
    print("oxDNA2 PyTorch Energy Test Script")
    print("=" * 80)
    print()

    # Read topology
    print(f"Reading topology from: {topology_file}")
    n_nucleotides, n_strands, base_types, strand_ids, n3_neighbors, n5_neighbors = read_topology(topology_file)
    print(f"  Number of nucleotides: {n_nucleotides}")
    print(f"  Number of strands: {n_strands}")
    print()

    # Read configuration
    print(f"Reading configuration from: {config_file}")
    time, box, energy, positions, orientations, velocities, angular_velocities = read_configuration(
        config_file, n_nucleotides)
    print(f"  Time: {time}")
    print(f"  Box: {box}")
    print(f"  Energy (total, potential, kinetic): {energy}")
    print()

    # Initialize energy model
    print("Initializing oxDNA2 energy model...")
    # Use default parameters (T=0.1, salt=0.5M)
    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)
    print()

    # Compute energies
    print("Computing energies using PyTorch implementation...")
    computed_energies = compute_all_pairwise_energies(
        model, positions, orientations, base_types, n3_neighbors, n5_neighbors)
    print()

    # Compare with reference files
    print("=" * 80)
    print("ENERGY COMPARISON RESULTS")
    print("=" * 80)
    print()

    # Map component names to JSON file names
    component_files = {
        'FENE': 'output_FENE.json',
        'BEXC': 'output_BEXC.json',
        'STCK': 'output_STCK.json',
        'NEXC': 'output_NEXC.json',
        'HB': 'output_HB.json',
        'CRSTCK': 'output_CRSTCK.json',
        'CXSTCK': 'output_CXSTCK.json',
        'total': 'output_total.json'
    }

    all_results = {}

    for component, filename in component_files.items():
        json_file = example_dir / filename

        if not json_file.exists():
            print(f"Warning: Reference file {filename} not found, skipping {component}")
            continue

        # Read reference
        ref_name, ref_energies = read_reference_energies(json_file)

        # Get computed energies
        comp_energies = computed_energies[component]

        # Compare
        errors, rmse, max_error, mean_error = compare_energies(comp_energies, ref_energies, component)

        all_results[component] = {
            'rmse': rmse,
            'max_error': max_error,
            'mean_error': mean_error,
            'errors': errors,
            'computed': comp_energies,
            'reference': ref_energies
        }

        # Print summary
        print(f"{component:10s} ({ref_name})")
        print(f"  RMSE:       {rmse:.6e}")
        print(f"  Max Error:  {max_error:.6e}")
        print(f"  Mean Error: {mean_error:.6e}")
        print()

    # Detailed per-nucleotide errors
    print("=" * 80)
    print("PER-NUCLEOTIDE ERROR BREAKDOWN")
    print("=" * 80)
    print()

    header = f"{'Nuc':>4s} "
    for component in component_files.keys():
        if component in all_results:
            header += f"{component:>12s} "
    print(header)
    print("-" * len(header))

    for i in range(n_nucleotides):
        row = f"{i:4d} "
        for component in component_files.keys():
            if component in all_results:
                error = all_results[component]['errors'][i]
                row += f"{error:12.6e} "
        print(row)

    print()

    # Detailed comparison table
    print("=" * 80)
    print("DETAILED ENERGY COMPARISON (Computed vs Reference)")
    print("=" * 80)
    print()

    for component, filename in component_files.items():
        if component not in all_results:
            continue

        result = all_results[component]
        print(f"{component} Energy:")
        print(f"{'Nuc':>4s} {'Computed':>15s} {'Reference':>15s} {'Error':>15s}")
        print("-" * 52)
        for i in range(n_nucleotides):
            comp = result['computed'][i]
            ref = result['reference'][i]
            err = result['errors'][i]
            print(f"{i:4d} {comp:15.8f} {ref:15.8f} {err:15.8e}")
        print()

    # Overall summary
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print()

    if 'total' in all_results:
        total_result = all_results['total']
        total_computed = np.sum(total_result['computed'])
        total_reference = np.sum(total_result['reference'])
        total_error = abs(total_computed - total_reference)

        print(f"Total Energy (Computed):  {total_computed:.10f} oxDNA su")
        print(f"Total Energy (Reference): {total_reference:.10f} oxDNA su")
        print(f"Total Error:              {total_error:.6e} oxDNA su")
        print(f"Relative Error:           {(total_error/abs(total_reference)*100):.4f}%")

    print()
    print("Test complete!")
    print()


if __name__ == "__main__":
    main()
