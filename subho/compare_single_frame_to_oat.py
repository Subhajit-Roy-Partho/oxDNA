#!/usr/bin/env python3
"""
Compare oxDNA2 PyTorch energies against oxDNA CPU ground truth on one frame.

This script:
1. Reads one oxDNA topology and one single-frame configuration (.dat)
2. Evaluates the current Torch oxDNA2 implementation on that frame
3. Runs `oat output_bonds` on the same frame to get oxDNA CPU pair-energy terms
4. Converts the `output_bonds` per-nucleotide arrays from raw double-counted
   accounting to split accounting
5. Reports term-by-term per-nucleotide errors and pair-sum differences

Important:
- `oat output_bonds -v` writes per-nucleotide arrays where every pair energy is
  added to both nucleotides.
- `oxDNA2Energy.compute_system_energies()` returns per-nucleotide arrays with
  split accounting, i.e. half of each pair contribution goes to each endpoint.
- For a fair comparison, this script divides the oat arrays by 2 before
  comparing them against Torch per-nucleotide outputs.

Example:
  python subho/compare_single_frame_to_oat.py
  python subho/compare_single_frame_to_oat.py \
      --top subho/example/dsdna8.top \
      --frame subho/example/init.dat \
      --temperature 20C \
      --salt 0.5 \
      --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from oxdna2_pytorch import oxDNA2Energy


TERM_ORDER = ["FENE", "BEXC", "STCK", "NEXC", "HB", "CRSTCK", "CXSTCK", "DH", "total"]


def read_topology(topology_file: Path):
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
            base_types.append(base_to_number[parts[1]])
            n3_neighbors.append(int(parts[2]))
            n5_neighbors.append(int(parts[3]))

    return (
        n_nucleotides,
        torch.tensor(base_types, dtype=torch.long),
        torch.tensor(n3_neighbors, dtype=torch.long),
        torch.tensor(n5_neighbors, dtype=torch.long),
    )


def read_configuration(config_file: Path, n_nucleotides: int):
    """Read oxDNA .dat configuration and return (positions, orientations)."""
    with config_file.open("r", encoding="utf-8") as f:
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
            a2 = np.cross(a3, a1)
            orientation = np.column_stack([a1, a2, a3]).astype(np.float32)
            positions.append(cm)
            orientations.append(orientation)

    return torch.tensor(np.array(positions)), torch.tensor(np.array(orientations))


def parse_temperature_token(token: str) -> float:
    """Match oxDNA temperature parsing for plain SU, K, or C values."""
    token = token.strip()
    if not token:
        raise ValueError("temperature token must not be empty")

    suffix = token[-1]
    if suffix in ("c", "C"):
        return (float(token[:-1]) + 273.15) * 0.1 / 300.0
    if suffix in ("k", "K"):
        return float(token[:-1]) * 0.1 / 300.0
    return float(token)


def resolve_device(device_name: str) -> torch.device:
    """Resolve requested Torch device with graceful fallback."""
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def default_seq_dep_file() -> Path:
    return Path(__file__).resolve().parent.parent / "oxDNA2_sequence_dependent_parameters.txt"


def write_single_frame_input(
    input_path: Path,
    top_path: Path,
    frame_path: Path,
    temperature_token: str,
    salt_concentration: float,
    use_average_seq: bool,
    seq_dep_file: Path | None,
) -> None:
    """Write an oxDNA input file for one-frame `oat output_bonds` analysis."""
    lines = [
        "backend = CPU",
        "steps = 1",
        "newtonian_steps = 103",
        "diff_coeff = 2.50",
        "thermostat = john",
        f"T = {temperature_token}",
        "dt = 0.005",
        "verlet_skin = 0.05",
        "interaction_type = DNA2",
        f"salt_concentration = {salt_concentration}",
        f"use_average_seq = {'true' if use_average_seq else 'false'}",
    ]
    if not use_average_seq:
        if seq_dep_file is None:
            raise ValueError("seq_dep_file is required when use_average_seq is false")
        lines.append(f"seq_dep_file = {seq_dep_file}")

    lines.extend(
        [
            f"topology = {top_path}",
            f"conf_file = {frame_path}",
            f"trajectory_file = {frame_path}",
            f"log_file = {input_path.parent / 'oat.log'}",
            "no_stdout_energy = 1",
            "restart_step_counter = 1",
            f"energy_file = {input_path.parent / 'energy.dat'}",
            "print_conf_interval = 1",
            "print_energy_every = 1",
            "time_scale = linear",
            "external_forces = 0",
            "refresh_vel = 1",
        ]
    )
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_local_oxpy_python_path() -> Path | None:
    """Find a locally built oxpy python path that contains oxpy/core.so."""
    repo_root = Path(__file__).resolve().parent.parent
    for core_so in repo_root.glob("build/**/oxpy/core.so"):
        return core_so.parent.parent
    return None


def find_local_analysis_build_path() -> Path | None:
    """Find a locally built analysis package path that contains get_confs."""
    repo_root = Path(__file__).resolve().parent.parent
    for get_confs_so in repo_root.glob("analysis/build/**/oxDNA_analysis_tools/UTILS/get_confs*.so"):
        return get_confs_so.parents[2]
    return None


def resolve_output_bonds_runner(oat_bin: str):
    """
    Resolve how to run output_bonds.

    Preferred path:
    - `oat output_bonds ...`

    Fallback path:
    - `/usr/bin/python3 analysis/src/oxDNA_analysis_tools/output_bonds.py ...`
      with PYTHONPATH pointing at the local oxpy build and analysis sources
    """
    oat_resolved = shutil.which(oat_bin)
    if oat_resolved is not None:
        return [oat_resolved, "output_bonds"], None, f"oat:{oat_resolved}"
    explicit_oat = Path(oat_bin)
    if explicit_oat.exists():
        return [str(explicit_oat.resolve()), "output_bonds"], None, f"oat:{explicit_oat.resolve()}"

    repo_root = Path(__file__).resolve().parent.parent
    python39 = Path("/usr/bin/python3")
    output_bonds_script = repo_root / "analysis" / "src" / "oxDNA_analysis_tools" / "output_bonds.py"
    oxpy_python_path = find_local_oxpy_python_path()
    analysis_build_path = find_local_analysis_build_path()
    if (
        python39.exists()
        and output_bonds_script.exists()
        and oxpy_python_path is not None
        and analysis_build_path is not None
    ):
        env = os.environ.copy()
        py_paths = [
            str(analysis_build_path),
            str(repo_root / "subho" / "compat"),
            str(oxpy_python_path),
            str(repo_root / "src"),
            str(repo_root / "analysis" / "src"),
        ]
        if env.get("PYTHONPATH"):
            py_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(py_paths)
        return [str(python39), str(output_bonds_script)], env, f"python:{python39}"

    raise RuntimeError(
        "Could not resolve an output_bonds runner. Neither `oat` nor a local Python 3.9 + oxpy build "
        "was available."
    )


def run_output_bonds(oat_bin: str, input_path: Path, frame_path: Path, output_prefix: Path):
    """Run output_bonds and return a description of the runner used."""
    runner_cmd, runner_env, runner_desc = resolve_output_bonds_runner(oat_bin)
    cmd = runner_cmd + [str(input_path), str(frame_path), "-v", str(output_prefix), "-q"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=runner_env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "output_bonds runner failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    return runner_desc


def load_oat_reference(output_prefix: Path) -> Dict[str, np.ndarray]:
    """Load raw per-nucleotide arrays written by `oat output_bonds -v`."""
    refs: Dict[str, np.ndarray] = {}
    for term in TERM_ORDER:
        json_path = output_prefix.with_name(f"{output_prefix.name}_{term}.json")
        if not json_path.exists():
            raise FileNotFoundError(f"Expected oat output file not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        key = next(iter(data.keys()))
        refs[term] = np.array(data[key], dtype=np.float64)
    return refs


def term_metrics(torch_split: np.ndarray, oat_raw: np.ndarray, torch_pair_sum: float) -> Dict[str, float]:
    """Compute split-accounting comparison metrics for one term."""
    oat_split = oat_raw / 2.0
    errors = torch_split - oat_split
    abs_errors = np.abs(errors)
    oat_pair_sum = float(oat_split.sum())
    sum_abs_diff = abs(torch_pair_sum - oat_pair_sum)
    rel_sum_diff_pct = 0.0 if math.isclose(oat_pair_sum, 0.0) else 100.0 * sum_abs_diff / abs(oat_pair_sum)

    return {
        "oat_pair_sum": oat_pair_sum,
        "torch_pair_sum": torch_pair_sum,
        "sum_abs_diff": sum_abs_diff,
        "rel_sum_diff_pct": rel_sum_diff_pct,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_error": float(abs_errors.max()),
        "mean_abs_error": float(abs_errors.mean()),
        "raw_accounting_max_abs_error": float(np.abs((2.0 * torch_split) - oat_raw).max()),
    }


def write_report_json(report_path: Path, payload: dict) -> None:
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one DNA2 frame against oat output_bonds ground truth")
    parser.add_argument("--top", default="subho/example/dsdna8.top", help="Path to topology (.top)")
    parser.add_argument("--frame", default="subho/example/init.dat", help="Path to single-frame oxDNA .dat file")
    parser.add_argument(
        "--temperature",
        default="20C",
        help="Temperature token for oxDNA and Torch (examples: 20C, 300K, 0.1)",
    )
    parser.add_argument("--salt", type=float, default=0.5, help="Salt concentration in M")
    parser.add_argument(
        "--use-average-seq",
        action="store_true",
        help="Use sequence-averaged DNA2 parameters instead of oxDNA2_sequence_dependent_parameters.txt",
    )
    parser.add_argument(
        "--seq-dep-file",
        default=None,
        help="Path to oxDNA2 sequence-dependent parameter file. Used when --use-average-seq is not set.",
    )
    parser.add_argument(
        "--compute-dtype",
        default="auto",
        choices=["auto", "float16", "float32", "float64", "float8_e4m3fn", "float8_e5m2"],
        help="Internal compute precision for oxDNA2Energy. Float8 modes quantize inputs and compute in float16.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device")
    parser.add_argument("--oat-bin", default="oat", help="Path to oat executable")
    parser.add_argument("--report-json", default=None, help="Optional path to write the full comparison report as JSON")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary oat input/output directory")
    args = parser.parse_args()

    top_path = Path(args.top).resolve()
    frame_path = Path(args.frame).resolve()
    if not top_path.exists():
        raise FileNotFoundError(f"Topology file not found: {top_path}")
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame file not found: {frame_path}")

    temperature_su = parse_temperature_token(args.temperature)
    seq_dep_file = None
    if not args.use_average_seq:
        seq_dep_file = Path(args.seq_dep_file).resolve() if args.seq_dep_file else default_seq_dep_file().resolve()
        if not seq_dep_file.exists():
            raise FileNotFoundError(f"Sequence-dependent parameter file not found: {seq_dep_file}")

    device = resolve_device(args.device)
    n_nucleotides, base_types, n3_neighbors, n5_neighbors = read_topology(top_path)
    positions, orientations = read_configuration(frame_path, n_nucleotides)

    positions = positions.to(device)
    orientations = orientations.to(device)
    base_types = base_types.to(device)
    n3_neighbors = n3_neighbors.to(device)
    n5_neighbors = n5_neighbors.to(device)

    model = oxDNA2Energy(
        temperature=temperature_su,
        salt_concentration=args.salt,
        use_average_seq=args.use_average_seq,
        seq_dep_file=None if seq_dep_file is None else str(seq_dep_file),
        grooving=True,
        compute_dtype=args.compute_dtype,
    ).to(device)

    torch_result = model.compute_system_energies(
        positions=positions,
        orientations=orientations,
        base_types=base_types,
        n3_neighbors=n3_neighbors,
        n5_neighbors=n5_neighbors,
    )

    temp_dir_obj = None
    if args.keep_temp:
        tmpdir = Path(tempfile.mkdtemp(prefix="torchdna_oat_compare_"))
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="torchdna_oat_compare_")
        tmpdir = Path(temp_dir_obj.name)

    try:
        input_path = tmpdir / "input_single_frame_dna2"
        output_prefix = tmpdir / "oat_ref"
        write_single_frame_input(
            input_path=input_path,
            top_path=top_path,
            frame_path=frame_path,
            temperature_token=args.temperature,
            salt_concentration=args.salt,
            use_average_seq=args.use_average_seq,
            seq_dep_file=seq_dep_file,
        )
        runner_desc = run_output_bonds(args.oat_bin, input_path, frame_path, output_prefix)
        oat_reference = load_oat_reference(output_prefix)

        torch_split_terms = {
            term: torch_result["per_nucleotide_terms"][term].detach().cpu().numpy().astype(np.float64)
            for term in TERM_ORDER
            if term != "total"
        }
        torch_split_terms["total"] = (
            torch_result["per_nucleotide_total"].detach().cpu().numpy().astype(np.float64)
        )

        torch_pair_sums = {
            term: float(torch_result["term_sums"][term].detach().cpu().item())
            for term in TERM_ORDER
            if term != "total"
        }
        torch_pair_sums["total"] = float(torch_result["total_sum"].detach().cpu().item())

        metrics = {}
        for term in TERM_ORDER:
            metrics[term] = term_metrics(torch_split_terms[term], oat_reference[term], torch_pair_sums[term])

        print("=" * 96)
        print("Single-frame oxDNA2 comparison: Torch vs oat output_bonds")
        print("=" * 96)
        print(f"Topology:            {top_path}")
        print(f"Frame:               {frame_path}")
        print(f"N nucleotides:       {n_nucleotides}")
        print(f"Device:              {device}")
        print(f"Temperature token:   {args.temperature}")
        print(f"Temperature (SU):    {temperature_su:.10f}")
        print(f"Salt (M):            {args.salt}")
        print(f"Average sequence:    {args.use_average_seq}")
        if not args.use_average_seq:
            print(f"Seq dep file:        {seq_dep_file}")
        print(f"Output runner:       {runner_desc}")
        print(f"Temporary dir:       {tmpdir}")
        print()
        print("Accounting:")
        print("  oat output_bonds -v arrays are raw per-nucleotide overlays (full pair energy added to both endpoints).")
        print("  This script divides those arrays by 2 before comparing to Torch per-nucleotide split accounting.")
        print()
        print(
            f"{'Term':10s} {'TorchPairSum':>14s} {'OatPairSum':>14s} {'RelDiff%':>10s} "
            f"{'RMSE':>12s} {'MaxAbs':>12s} {'MeanAbs':>12s}"
        )
        print("-" * 96)
        for term in TERM_ORDER:
            m = metrics[term]
            print(
                f"{term:10s} {m['torch_pair_sum']:14.8f} {m['oat_pair_sum']:14.8f} "
                f"{m['rel_sum_diff_pct']:10.6f} {m['rmse']:12.6e} "
                f"{m['max_abs_error']:12.6e} {m['mean_abs_error']:12.6e}"
            )

        if args.report_json:
            payload = {
                "topology": str(top_path),
                "frame": str(frame_path),
                "device": str(device),
                "compute_dtype": args.compute_dtype,
                "temperature_token": args.temperature,
                "temperature_simulation_units": temperature_su,
                "salt_concentration": args.salt,
                "use_average_seq": args.use_average_seq,
                "seq_dep_file": None if seq_dep_file is None else str(seq_dep_file),
                "output_runner": runner_desc,
                "tmpdir": str(tmpdir),
                "metrics": metrics,
            }
            write_report_json(Path(args.report_json).resolve(), payload)
            print()
            print(f"Report JSON written to: {Path(args.report_json).resolve()}")
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
