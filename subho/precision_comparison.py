#!/usr/bin/env python3
"""
Precision comparison: PyTorch oxDNA2 vs C++ reference (via oxpy).

For each test case the script:
  1. Uses oxpy pair_interaction_term to compute per-nucleotide per-term
     energies from the C++ reference implementation.
  2. Evaluates our PyTorch implementation at float16, float32, and float64.
  3. Prints a side-by-side table of sums + per-nucleotide RMSE / max-abs for
     every term and every precision.

Term IDs (DNA2Interaction enum):
  0 BACKBONE (FENE)  1 BONDED_EX_VOL  2 STACKING
  3 NONBONDED_EX_VOL 4 HYDROGEN_BONDING 5 CROSS_STACKING
  6 COAXIAL_STACKING  7 DEBYE_HUCKEL
"""

from __future__ import annotations

import os
import sys
import math
import tempfile
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ── locate oxpy built in this repo ──────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_OXPY_BUILD = _REPO_ROOT / "build" / "python"
if str(_OXPY_BUILD) not in sys.path:
    sys.path.insert(0, str(_OXPY_BUILD))

import oxpy                               # noqa: E402  (local build)

# ── locate our PyTorch module ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oxdna2_pytorch import oxDNA2Energy   # noqa: E402


# ── DNA2 interaction term IDs (from DNAInteraction.h / DNA2Interaction.h) ───
TERM_IDS: Dict[str, int] = {
    "FENE":   0,   # backbone (FENE)
    "BEXC":   1,   # bonded excluded volume
    "STCK":   2,   # stacking
    "NEXC":   3,   # non-bonded excluded volume
    "HB":     4,   # hydrogen bonding
    "CRSTCK": 5,   # cross-stacking
    "CXSTCK": 6,   # coaxial stacking
    "DH":     7,   # Debye-Hückel
}
TERMS = list(TERM_IDS.keys())

PRECISIONS = ["float16", "float32", "float64"]


# ── helper: temperature token → simulation units ────────────────────────────
def temperature_to_su(token: str) -> float:
    token = token.strip()
    if token[-1] in ("c", "C"):
        return (float(token[:-1]) + 273.15) * 0.1 / 300.0
    if token[-1] in ("k", "K"):
        return float(token[:-1]) * 0.1 / 300.0
    return float(token)


# ── topology / configuration readers ────────────────────────────────────────
def read_topology(top: Path):
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    with top.open() as f:
        n = int(f.readline().split()[0])
        base_types, n3, n5 = [], [], []
        for line in f:
            p = line.split()
            base_types.append(base_map[p[1]])
            n3.append(int(p[2]))
            n5.append(int(p[3]))
    return n, base_types, n3, n5


def read_configuration(conf: Path, n: int):
    with conf.open() as f:
        f.readline(); f.readline(); f.readline()
        positions, orientations = [], []
        for _ in range(n):
            v = list(map(float, f.readline().split()))
            cm = np.array(v[0:3], dtype=np.float64)
            a1 = np.array(v[3:6], dtype=np.float64)
            a3 = np.array(v[6:9], dtype=np.float64)
            a2 = np.cross(a3, a1)
            positions.append(cm)
            orientations.append(np.column_stack([a1, a2, a3]))
    return np.array(positions), np.array(orientations)


# ── write a minimal DNA2 input file for oxpy ────────────────────────────────
def write_oxpy_input(
    dest: Path,
    top: Path,
    conf: Path,
    temperature_token: str,
    salt: float,
    seq_dep_file: Path,
) -> None:
    lines = [
        "backend = CPU",
        "interaction_type = DNA2",
        "steps = 0",
        "newtonian_steps = 103",
        "diff_coeff = 2.50",
        "thermostat = john",
        f"T = {temperature_token}",
        "dt = 0.005",
        "verlet_skin = 0.05",
        f"salt_concentration = {salt}",
        f"seq_dep_file = {seq_dep_file}",
        "use_average_seq = 0",
        f"topology = {top}",
        f"conf_file = {conf}",
        f"trajectory_file = {dest.parent / 'traj.dat'}",
        "no_stdout_energy = 1",
        "restart_step_counter = 1",
        "energy_file = /dev/null",
        "print_conf_interval = 1",
        "print_energy_every = 1",
        "time_scale = linear",
        "external_forces = 0",
        "refresh_vel = 1",
    ]
    dest.write_text("\n".join(lines) + "\n")


# ── C++ reference via oxpy ───────────────────────────────────────────────────
def compute_reference_oxpy(
    top: Path,
    conf: Path,
    temperature_token: str,
    salt: float,
    seq_dep_file: Path,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Return per-nucleotide energies with split accounting (half to each endpoint)
    computed by the C++ reference via oxpy pair_interaction_term.
    """
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input_dna2"
        write_oxpy_input(inp, top, conf, temperature_token, salt, seq_dep_file)

        with oxpy.Context():
            m = oxpy.OxpyManager(str(inp))
            ci = m.config_info()
            inter = ci.interaction
            particles = ci.particles()
            n = len(particles)

            per_nuc = {t: np.zeros(n, dtype=np.float64) for t in TERMS}
            per_nuc["total"] = np.zeros(n, dtype=np.float64)

            for i in range(n):
                for j in range(i + 1, n):
                    p, q = particles[i], particles[j]
                    pair_total = 0.0
                    for term, tid in TERM_IDS.items():
                        e = inter.pair_interaction_term(tid, p, q)
                        per_nuc[term][i] += 0.5 * e
                        per_nuc[term][j] += 0.5 * e
                        pair_total += e
                    per_nuc["total"][i] += 0.5 * pair_total
                    per_nuc["total"][j] += 0.5 * pair_total

    return per_nuc, n


# ── PyTorch evaluation at a given precision ──────────────────────────────────
def compute_torch(
    positions: np.ndarray,
    orientations: np.ndarray,
    base_types: List[int],
    n3: List[int],
    n5: List[int],
    temperature_su: float,
    salt: float,
    seq_dep_file: Path,
    compute_dtype: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    pos_t = torch.tensor(positions, dtype=torch.float64).to(device)
    ori_t = torch.tensor(orientations, dtype=torch.float64).to(device)
    bt_t  = torch.tensor(base_types, dtype=torch.long).to(device)
    n3_t  = torch.tensor(n3, dtype=torch.long).to(device)
    n5_t  = torch.tensor(n5, dtype=torch.long).to(device)

    model = oxDNA2Energy(
        temperature=temperature_su,
        salt_concentration=salt,
        use_average_seq=False,
        seq_dep_file=str(seq_dep_file),
        grooving=True,
        compute_dtype=compute_dtype,
    ).to(device)

    with torch.no_grad():
        result = model.compute_system_energies(pos_t, ori_t, bt_t, n3_t, n5_t)

    out: Dict[str, np.ndarray] = {}
    for term in TERMS:
        out[term] = result["per_nucleotide_terms"][term].cpu().double().numpy()
    out["total"] = result["per_nucleotide_total"].cpu().double().numpy()
    return out


# ── metrics ──────────────────────────────────────────────────────────────────
def metrics(torch_arr: np.ndarray, ref_arr: np.ndarray):
    diff = torch_arr - ref_arr
    return {
        "torch_sum": float(torch_arr.sum()),
        "ref_sum":   float(ref_arr.sum()),
        "sum_diff":  float(torch_arr.sum() - ref_arr.sum()),
        "rmse":      float(np.sqrt(np.mean(diff ** 2))),
        "max_abs":   float(np.abs(diff).max()),
    }


# ── pretty printer ───────────────────────────────────────────────────────────
def print_case(
    name: str,
    n: int,
    ref: Dict[str, np.ndarray],
    torch_results: Dict[str, Dict[str, np.ndarray]],
) -> None:
    all_terms = TERMS + ["total"]
    header_width = 130

    print("=" * header_width)
    print(f"  CASE: {name}   (N={n} nucleotides)")
    print("=" * header_width)

    # ── per-term table ──────────────────────────────────────────────────────
    print(f"\n{'Term':8s}  {'Ref sum':>14s}", end="")
    for prec in PRECISIONS:
        print(f"  {prec:>10s}_sum  {prec:>10s}_Δsum  {prec:>10s}_rmse  {prec:>10s}_max", end="")
    print()
    print("-" * header_width)

    for term in all_terms:
        ref_sum = float(ref[term].sum())
        print(f"{term:8s}  {ref_sum:14.8f}", end="")
        for prec in PRECISIONS:
            m = metrics(torch_results[prec][term], ref[term])
            print(
                f"  {m['torch_sum']:14.8f}  {m['sum_diff']:+12.6e}"
                f"  {m['rmse']:12.6e}  {m['max_abs']:12.6e}",
                end="",
            )
        print()

    # ── precision-ranked summary ────────────────────────────────────────────
    print(f"\n  Total energy sum comparison:")
    ref_total = float(ref["total"].sum())
    print(f"    C++ reference : {ref_total:.10f}")
    for prec in PRECISIONS:
        ts = float(torch_results[prec]["total"].sum())
        print(f"    {prec:10s}    : {ts:.10f}   (Δ = {ts - ref_total:+.4e})")
    print()


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side precision comparison: PyTorch oxDNA2 vs C++ reference"
    )
    parser.add_argument("--temperature", default="20C",
                        help="Temperature token, e.g. 20C 300K 0.1 (default: 20C)")
    parser.add_argument("--salt", type=float, default=0.5,
                        help="Salt concentration in M (default: 0.5)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device: cpu / cuda / mps (default: cpu)")
    parser.add_argument("--seq-dep-file", default=None,
                        help="Path to oxDNA2 sequence-dependent parameter file")
    args = parser.parse_args()

    seq_dep_file = (
        Path(args.seq_dep_file).resolve()
        if args.seq_dep_file
        else (_REPO_ROOT / "oxDNA2_sequence_dependent_parameters.txt").resolve()
    )
    if not seq_dep_file.exists():
        sys.exit(f"Sequence-dependent parameter file not found: {seq_dep_file}")

    temperature_su = temperature_to_su(args.temperature)
    device = torch.device(args.device)

    # ── test cases ──────────────────────────────────────────────────────────
    cases = [
        {
            "name": "dsDNA 8bp (16 nt)",
            "top":  _REPO_ROOT / "subho" / "example" / "dsdna8.top",
            "conf": _REPO_ROOT / "subho" / "example" / "init.dat",
        },
        {
            "name": "Hairpin (18 nt)",
            "top":  _REPO_ROOT / "examples" / "HAIRPIN" / "initial.top",
            "conf": _REPO_ROOT / "examples" / "HAIRPIN" / "initial.conf",
        },
    ]

    print(f"\noxDNA2 Precision Comparison: PyTorch vs C++ Reference")
    print(f"Temperature : {args.temperature}  ({temperature_su:.8f} SU)")
    print(f"Salt        : {args.salt} M")
    print(f"Seq-dep file: {seq_dep_file}")
    print(f"Torch device: {device}")
    print(f"Precisions  : {PRECISIONS}")
    print()

    for case in cases:
        top  = Path(case["top"]).resolve()
        conf = Path(case["conf"]).resolve()

        if not top.exists():
            print(f"[SKIP] {case['name']}: topology not found at {top}\n")
            continue
        if not conf.exists():
            print(f"[SKIP] {case['name']}: conf not found at {conf}\n")
            continue

        print(f"Loading {case['name']} ...")
        n, base_types, n3, n5 = read_topology(top)
        positions, orientations = read_configuration(conf, n)

        print(f"  Computing C++ reference via oxpy ...")
        try:
            ref, _ = compute_reference_oxpy(top, conf, args.temperature, args.salt, seq_dep_file)
        except Exception as exc:
            print(f"  [SKIP] C++ reference failed: {exc}\n")
            continue

        torch_results: Dict[str, Dict[str, np.ndarray]] = {}
        for prec in PRECISIONS:
            print(f"  Computing PyTorch ({prec}) ...")
            try:
                torch_results[prec] = compute_torch(
                    positions, orientations, base_types, n3, n5,
                    temperature_su, args.salt, seq_dep_file,
                    prec, device,
                )
            except Exception as exc:
                print(f"    ERROR for {prec}: {exc}")
                # fill with NaN so the table still prints
                nan_arr = np.full(n, float("nan"))
                torch_results[prec] = {t: nan_arr.copy() for t in TERMS + ["total"]}

        print_case(case["name"], n, ref, torch_results)


if __name__ == "__main__":
    main()
