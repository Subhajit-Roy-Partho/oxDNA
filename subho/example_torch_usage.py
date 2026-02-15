#!/usr/bin/env python3
"""
Example: using oxDNA2Energy with PyTorch on a real oxDNA configuration.

This script:
1. Loads an oxDNA topology (.top) and configuration (.dat)
2. Computes term-by-term energies with oxDNA2Energy
   using the fully vectorized system path (no Python pair loops)
4. Prints system totals and per-nucleotide split energies

Run:
  python subho/example_torch_usage.py
  python subho/example_torch_usage.py --top subho/example/dsdna8.top --conf subho/example/init.dat
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from oxdna2_pytorch import oxDNA2Energy


def read_topology(topology_file: Path) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read oxDNA topology and return (N, base_types, n3_neighbors, n5_neighbors)."""
    base_to_number = {"A": 0, "C": 1, "G": 2, "T": 3}

    with topology_file.open("r", encoding="utf-8") as f:
        first_line = f.readline().split()
        n_nucleotides = int(first_line[0])

        base_types = []
        n3_neighbors = []
        n5_neighbors = []

        for line in f:
            parts = line.split()
            base = parts[1]
            n3 = int(parts[2])
            n5 = int(parts[3])

            base_types.append(base_to_number[base])
            n3_neighbors.append(n3)
            n5_neighbors.append(n5)

    return (
        n_nucleotides,
        torch.tensor(base_types, dtype=torch.long),
        torch.tensor(n3_neighbors, dtype=torch.long),
        torch.tensor(n5_neighbors, dtype=torch.long),
    )


def read_configuration(config_file: Path, n_nucleotides: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read oxDNA .dat configuration and return (positions, orientations)."""
    with config_file.open("r", encoding="utf-8") as f:
        # Skip 3 header lines: time, box, energies
        f.readline()
        f.readline()
        f.readline()

        positions = []
        orientations = []
        for _ in range(n_nucleotides):
            vals = list(map(float, f.readline().split()))
            cm = np.array(vals[0:3], dtype=np.float32)
            a1 = np.array(vals[3:6], dtype=np.float32)
            a3 = np.array(vals[6:9], dtype=np.float32)

            # oxDNA convention: a2 = a3 x a1
            a2 = np.cross(a3, a1)
            orientation = np.column_stack([a1, a2, a3]).astype(np.float32)

            positions.append(cm)
            orientations.append(orientation)

    return torch.tensor(np.array(positions)), torch.tensor(np.array(orientations))


def main() -> None:
    parser = argparse.ArgumentParser(description="Example torch usage for oxDNA2Energy")
    parser.add_argument("--top", default="subho/example/dsdna8.top", help="Path to topology (.top)")
    parser.add_argument("--conf", default="subho/example/init.dat", help="Path to configuration (.dat)")
    parser.add_argument("--temperature", type=float, default=0.1, help="oxDNA simulation temperature")
    parser.add_argument("--salt", type=float, default=0.5, help="Salt concentration in M")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device")
    args = parser.parse_args()

    top_path = Path(args.top)
    conf_path = Path(args.conf)
    if not top_path.exists():
        raise FileNotFoundError(f"Topology file not found: {top_path}")
    if not conf_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")

    n, base_types, n3_neighbors, n5_neighbors = read_topology(top_path)
    positions, orientations = read_configuration(conf_path, n)

    positions = positions.to(device)
    orientations = orientations.to(device)
    base_types = base_types.to(device)
    n3_neighbors = n3_neighbors.to(device)
    n5_neighbors = n5_neighbors.to(device)

    model = oxDNA2Energy(temperature=args.temperature, salt_concentration=args.salt).to(device)
    result = model.compute_system_energies(
        positions=positions,
        orientations=orientations,
        base_types=base_types,
        n3_neighbors=n3_neighbors,
        n5_neighbors=n5_neighbors,
    )

    terms = result["pair_terms"]
    pair_total = result["pair_total"]
    per_nucleotide_total = result["per_nucleotide_total"]
    n_pairs = pair_total.numel()

    print("=" * 78)
    print("oxDNA2Energy torch example")
    print("=" * 78)
    print(f"Topology:       {top_path}")
    print(f"Configuration:  {conf_path}")
    print(f"N nucleotides:  {n}")
    print(f"N pairs (i<j):  {n_pairs}")
    print(f"Device:         {device}")
    print(f"T:              {args.temperature}")
    print(f"Salt (M):       {args.salt}")
    print()
    print("System energy by term (pairwise sums, oxDNA units):")
    for name, val in terms.items():
        print(f"  {name:7s}: {val.sum().item(): .8f}")
    print(f"  {'TOTAL':7s}: {pair_total.sum().item(): .8f}")
    print()
    print(
        f"Check split accounting: sum(per_nucleotide_total) = {per_nucleotide_total.sum().item(): .8f}, "
        f"pair_total = {pair_total.sum().item(): .8f}"
    )
    print()
    print("First 8 per-nucleotide TOTAL energies (split):")
    for i in range(min(8, n)):
        print(f"  i={i:2d}: {per_nucleotide_total[i].item(): .8f}")


if __name__ == "__main__":
    main()
