#!/usr/bin/env python3
"""
Generate a markdown comparison of oxDNA C++ ground truth vs Torch implementations.

Ground truth is taken from the native C++ `pair_energy` observable via the local
oxDNA analysis stack. The comparison covers:
- current `subho/oxdna2_pytorch.py` with `compute_dtype=float32`
- current `subho/oxdna2_pytorch.py` with `compute_dtype=float64`
- installed `../oxdna-torch`
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from oxdna_torch import OxDNAEnergy, read_configuration as pkg_read_configuration
from oxdna_torch import read_topology as pkg_read_topology

from compare_single_frame_to_oat import (
    parse_temperature_token,
    read_configuration,
    read_topology,
    resolve_output_bonds_runner,
    write_single_frame_input,
)
from oxdna2_pytorch import oxDNA2Energy

import subprocess


TERM_ORDER = ["FENE", "BEXC", "STCK", "NEXC", "HB", "CRSTCK", "CXSTCK", "DH", "total"]
SPEED_WARMUP_RUNS = 5
SPEED_MEASURED_RUNS = 20


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    structure_class: str
    top: str
    frame: str
    temperature_token: str
    salt: float
    use_average_seq: bool
    notes: str
    topology_format: str = "old"


CASES: List[Case] = [
    Case(
        name="dsdna8_duplex",
        title="DSDNA8 Duplex",
        structure_class="Duplex",
        top="test/DNA/DSDNA8/dsdna8.top",
        frame="test/DNA/DSDNA8/init.dat",
        temperature_token="20C",
        salt=0.5,
        use_average_seq=True,
        notes="Matches the DNA2 duplex validation frame used earlier.",
    ),
    Case(
        name="force_field_three_strand",
        title="FORCE_FIELD 3-Strand",
        structure_class="Three-Strand Complex",
        top="test/DNA/FORCE_FIELD/init.top",
        frame="test/DNA/FORCE_FIELD/init.dat",
        temperature_token="20C",
        salt=1.0,
        use_average_seq=True,
        notes=(
            "Source topology uses the newer sequence-style format. "
            "It is converted to old-format connectivity for both Torch implementations."
        ),
        topology_format="new",
    ),
    Case(
        name="ssdna15_md",
        title="SSDNA15 MD",
        structure_class="Single Strand",
        top="test/DNA/SSDNA15/MD/ssdna15.top",
        frame="test/DNA/SSDNA15/MD/init.dat",
        temperature_token="300K",
        salt=0.5,
        use_average_seq=True,
        notes=(
            "The frame comes from the legacy SSDNA15 test tree, "
            "but the comparison is run under oxDNA2 average-sequence settings."
        ),
    ),
]


def convert_new_topology_to_old(src: Path, dst: Path) -> Path:
    """Convert the newer sequence-style topology format to old per-nucleotide format."""
    lines = src.read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    n_nucleotides = int(header[0])
    n_strands = int(header[1])
    direction = header[2] if len(header) > 2 else "3->5"

    out_lines = [f"{n_nucleotides} {n_strands}"]
    offset = 0

    for strand_id in range(1, n_strands + 1):
        parts = lines[strand_id].split()
        sequence = parts[0].strip()
        circular = any(part.strip().lower() == "circular=true" for part in parts[1:])
        strand_len = len(sequence)

        for local_idx, base in enumerate(sequence):
            if direction == "5->3":
                n3 = offset + local_idx + 1 if local_idx + 1 < strand_len else (offset if circular else -1)
                n5 = offset + local_idx - 1 if local_idx - 1 >= 0 else (offset + strand_len - 1 if circular else -1)
            elif direction == "3->5":
                n3 = offset + local_idx - 1 if local_idx - 1 >= 0 else (offset + strand_len - 1 if circular else -1)
                n5 = offset + local_idx + 1 if local_idx + 1 < strand_len else (offset if circular else -1)
            else:
                raise ValueError(f"Unsupported topology direction token: {direction}")

            out_lines.append(f"{strand_id} {base} {n3} {n5}")

        offset += strand_len

    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return dst


def prepare_topology(case: Case, workdir: Path) -> Path:
    top_path = Path(case.top)
    if case.topology_format == "old":
        return top_path
    if case.topology_format == "new":
        return convert_new_topology_to_old(top_path, workdir / f"{case.name}.old.top")
    raise ValueError(f"Unknown topology_format: {case.topology_format}")


def load_cpp_pair_sums(top_path: Path, frame_path: Path, case: Case, workdir: Path) -> Dict[str, float]:
    """Evaluate the native C++ pair_energy observable and sum each component over all pairs."""
    input_path = workdir / f"{case.name}.input"
    data_file = workdir / f"{case.name}.pairs.dat"
    write_single_frame_input(
        input_path=input_path,
        top_path=top_path,
        frame_path=frame_path,
        temperature_token=case.temperature_token,
        salt_concentration=case.salt,
        use_average_seq=case.use_average_seq,
        seq_dep_file=None,
    )
    runner_cmd, runner_env, _ = resolve_output_bonds_runner("oat")
    cmd = runner_cmd + [str(input_path), str(frame_path), "-d", str(data_file), "-q"]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=runner_env)

    sums = {term: 0.0 for term in TERM_ORDER}
    for line in data_file.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        sums["FENE"] += float(parts[2])
        sums["BEXC"] += float(parts[3])
        sums["STCK"] += float(parts[4])
        sums["NEXC"] += float(parts[5])
        sums["HB"] += float(parts[6])
        sums["CRSTCK"] += float(parts[7])
        sums["CXSTCK"] += float(parts[8])
        sums["DH"] += float(parts[9])
        sums["total"] += float(parts[10])

    return sums


def current_impl_sums(top_path: Path, frame_path: Path, case: Case, compute_dtype: str) -> Dict[str, float]:
    n, base_types, n3_neighbors, n5_neighbors = read_topology(top_path)
    positions, orientations = read_configuration(frame_path, n)
    model = oxDNA2Energy(
        temperature=parse_temperature_token(case.temperature_token),
        salt_concentration=case.salt,
        use_average_seq=case.use_average_seq,
        seq_dep_file=None,
        grooving=True,
        compute_dtype=compute_dtype,
    )
    result = model.compute_system_energies(
        positions=positions,
        orientations=orientations,
        base_types=base_types,
        n3_neighbors=n3_neighbors,
        n5_neighbors=n5_neighbors,
    )
    sums = {term: float(result["term_sums"][term]) for term in result["term_sums"]}
    sums["total"] = float(result["total_sum"])
    return sums


def package_sums(top_path: Path, frame_path: Path, case: Case) -> Dict[str, float]:
    topology = pkg_read_topology(top_path)
    state = pkg_read_configuration(frame_path, topology)
    model = OxDNAEnergy(
        topology=topology,
        temperature=parse_temperature_token(case.temperature_token),
        seq_dependent=not case.use_average_seq,
        use_oxdna2=True,
        salt_concentration=case.salt,
    )
    comps = model.energy_components(state)
    sums = {
        "FENE": float(comps["fene"]),
        "BEXC": float(comps["bonded_excl_vol"]),
        "STCK": float(comps["stacking"]),
        "NEXC": float(comps["nonbonded_excl_vol"]),
        "HB": float(comps["hbond"]),
        "CRSTCK": float(comps["cross_stacking"]),
        "CXSTCK": float(comps["coaxial_stacking"]),
        "DH": float(comps["debye_huckel"]),
    }
    sums["total"] = sum(sums.values())
    return sums


def _synchronize_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark_callable(fn, device: str, warmup: int = SPEED_WARMUP_RUNS, repeats: int = SPEED_MEASURED_RUNS) -> float:
    """Return average milliseconds per invocation for a callable."""
    for _ in range(warmup):
        fn()
    _synchronize_device(device)

    elapsed = 0.0
    for _ in range(repeats):
        _synchronize_device(device)
        start = time.perf_counter()
        fn()
        _synchronize_device(device)
        elapsed += time.perf_counter() - start
    return elapsed * 1000.0 / repeats


def current_impl_speed(top_path: Path, frame_path: Path, case: Case, device: str) -> float:
    n, base_types, n3_neighbors, n5_neighbors = read_topology(top_path)
    positions, orientations = read_configuration(frame_path, n)
    positions = positions.to(device).requires_grad_(True)
    orientations = orientations.to(device).requires_grad_(True)
    base_types = base_types.to(device)
    n3_neighbors = n3_neighbors.to(device)
    n5_neighbors = n5_neighbors.to(device)

    model = oxDNA2Energy(
        temperature=parse_temperature_token(case.temperature_token),
        salt_concentration=case.salt,
        use_average_seq=case.use_average_seq,
        seq_dep_file=None,
        grooving=True,
        compute_dtype="float32",
    ).to(device)

    def run_once():
        return model.compute_system_energies(
            positions=positions,
            orientations=orientations,
            base_types=base_types,
            n3_neighbors=n3_neighbors,
            n5_neighbors=n5_neighbors,
        )

    return benchmark_callable(run_once, device)


def package_speed(top_path: Path, frame_path: Path, case: Case, device: str) -> float | None:
    if device == "mps":
        # This package currently materializes float64 tensors in paths that MPS cannot accept.
        try:
            topology = pkg_read_topology(top_path)
            state = pkg_read_configuration(frame_path, topology).to(torch.device("mps"))
            model = OxDNAEnergy(
                topology=topology,
                temperature=parse_temperature_token(case.temperature_token),
                seq_dependent=not case.use_average_seq,
                use_oxdna2=True,
                salt_concentration=case.salt,
            ).to("mps")
            _ = model(state)
        except Exception:
            return None

    topology = pkg_read_topology(top_path)
    state = pkg_read_configuration(frame_path, topology)
    if device != "cpu":
        state = state.to(torch.device(device))
    state = state.requires_grad_(True)
    model = OxDNAEnergy(
        topology=topology,
        temperature=parse_temperature_token(case.temperature_token),
        seq_dependent=not case.use_average_seq,
        use_oxdna2=True,
        salt_concentration=case.salt,
    )
    if device == "cpu":
        model = model.to(torch.float32)

    def run_once():
        return model(state)

    return benchmark_callable(run_once, device)


def error_pct(value: float, ref: float) -> float:
    return abs(value - ref) / abs(ref) * 100.0


def error_text(value: float, ref: float) -> str:
    diff = abs(value - ref)
    if abs(ref) <= 1e-12:
        return "0" if diff <= 1e-12 else f"abs {diff:.3e}"
    return f"{error_pct(value, ref):.6f}%"


def winner_for_term(ref: float, candidates: Dict[str, float]) -> str:
    diffs = {name: abs(value - ref) for name, value in candidates.items()}
    best_name = min(diffs, key=diffs.get)
    if list(diffs.values()).count(diffs[best_name]) > 1:
        return "tie"
    return best_name


def format_value(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.8f}"


def generate_markdown(results: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Comparison")
    lines.append("")
    lines.append("Ground truth is the native oxDNA C++ `pair_energy` observable queried through the local analysis stack.")
    lines.append("")
    lines.append("Compared implementations:")
    lines.append("- `current-f32`: `subho/oxdna2_pytorch.py` with `compute_dtype=\"float32\"`")
    lines.append("- `current-f64`: `subho/oxdna2_pytorch.py` with `compute_dtype=\"float64\"`")
    lines.append("- `oxdna_torch`: installed `../oxdna-torch` package")
    lines.append("")
    lines.append("All cases below are evaluated under oxDNA2 average-sequence settings so the three implementations are compared on the same force field.")
    lines.append("The speed table below measures a single differentiable forward pass with a small warm-up (`5` runs) and `20` timed iterations per case.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Class | N | Settings | Nonzero Terms | Current f32 Total Err | Current f64 Total Err | oxdna_torch Total Err | Closest Total |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---|")
    for result in results:
        total = result["term_rows"]["total"]
        lines.append(
            f"| {result['title']} | {result['class']} | {result['n']} | {result['settings']} | "
            f"{', '.join(result['nonzero_terms'])} | "
            f"{error_text(total['current_f32'], total['ref'])} | "
            f"{error_text(total['current_f64'], total['ref'])} | "
            f"{error_text(total['oxdna_torch'], total['ref'])} | "
            f"{total['winner']} |"
        )

    for result in results:
        lines.append("")
        lines.append(f"## {result['title']}")
        lines.append("")
        lines.append(f"- Source: `{result['source_top']}` and `{result['source_frame']}`")
        lines.append(f"- Settings: `{result['settings']}`")
        lines.append(f"- Note: {result['notes']}")
        lines.append("")
        lines.append("| Term | Ground Truth | Current float32 | Error | Current float64 | Error | oxdna_torch | Error | Closest |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for term in TERM_ORDER:
            row = result["term_rows"][term]
            lines.append(
                f"| `{term}` | {format_value(row['ref'])} | "
                f"{format_value(row['current_f32'])} | {error_text(row['current_f32'], row['ref'])} | "
                f"{format_value(row['current_f64'])} | {error_text(row['current_f64'], row['ref'])} | "
                f"{format_value(row['oxdna_torch'])} | {error_text(row['oxdna_torch'], row['ref'])} | "
                f"{row['winner']} |"
            )

    lines.append("")
    lines.append("## Speed")
    lines.append("")
    lines.append("Timings are average wall-clock milliseconds per forward pass. Current Torch timings use `compute_dtype=\"float32\"` with autograd-enabled inputs. `oxdna_torch` currently cannot run on MPS here because its float64-first path is rejected by Metal, so the GPU column is marked as unsupported.")
    lines.append("These are small test systems, so GPU launch overhead is part of the result and MPS can be slower than CPU on this workload.")
    lines.append("")
    lines.append("| Case | Current CPU (ms) | Current MPS (ms) | MPS speedup vs CPU | oxdna_torch CPU (ms) | Current CPU speedup vs oxdna_torch | oxdna_torch MPS |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for result in results:
        speed = result["speed"]
        current_cpu = speed["current_cpu_ms"]
        current_mps = speed["current_mps_ms"]
        package_cpu = speed["package_cpu_ms"]
        package_mps = speed["package_mps_ms"]
        mps_speedup = current_cpu / current_mps if current_mps and current_mps > 0 else float("nan")
        package_speedup = package_cpu / current_cpu if package_cpu and package_cpu > 0 else float("nan")
        lines.append(
            f"| {result['title']} | {current_cpu:.3f} | {current_mps:.3f} | {mps_speedup:.2f}x | "
            f"{package_cpu:.3f} | {package_speedup:.2f}x | {package_mps} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Relative errors are reported only when the C++ ground-truth term is nonzero.")
    lines.append("- For zero-reference terms, the table shows either `0` or an absolute deviation in oxDNA energy units.")
    lines.append("- The `FORCE_FIELD` case uses a temporary conversion from the newer sequence-style topology format to old-format connectivity so both PyTorch implementations can read the same frame.")
    lines.append("- `current-f64` is the tightest match overall in this repo when CPU double precision is available.")
    lines.append("- `oxdna_torch` does not currently provide a working MPS path on this machine without source changes because it materializes float64 tensors in MPS-bound code paths.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "comparison.md"
    results: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="torchdna_comparison_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for case in CASES:
            top_path = prepare_topology(case, tmpdir)
            frame_path = Path(case.frame)
            ref = load_cpp_pair_sums(top_path, frame_path, case, tmpdir)
            current_f32 = current_impl_sums(top_path, frame_path, case, "float32")
            current_f64 = current_impl_sums(top_path, frame_path, case, "float64")
            pkg = package_sums(top_path, frame_path, case)
            current_cpu_ms = current_impl_speed(top_path, frame_path, case, "cpu")
            current_mps_ms = current_impl_speed(top_path, frame_path, case, "mps")
            package_cpu_ms = package_speed(top_path, frame_path, case, "cpu")
            package_mps_ms = package_speed(top_path, frame_path, case, "mps")

            term_rows = {}
            for term in TERM_ORDER:
                candidates = {
                    "current-f32": current_f32[term],
                    "current-f64": current_f64[term],
                    "oxdna_torch": pkg[term],
                }
                term_rows[term] = {
                    "ref": ref[term],
                    "current_f32": current_f32[term],
                    "current_f64": current_f64[term],
                    "oxdna_torch": pkg[term],
                    "winner": winner_for_term(ref[term], candidates),
                }

            nonzero_terms = [term for term in TERM_ORDER if term != "total" and abs(ref[term]) > 1e-12]
            n_nucleotides, _, _, _ = read_topology(top_path)

            results.append(
                {
                    "title": case.title,
                    "class": case.structure_class,
                    "source_top": case.top,
                    "source_frame": case.frame,
                    "n": n_nucleotides,
                    "settings": f"DNA2 avg-seq, T={case.temperature_token}, salt={case.salt} M",
                    "nonzero_terms": nonzero_terms,
                    "notes": case.notes,
                    "term_rows": term_rows,
                    "speed": {
                        "current_cpu_ms": current_cpu_ms,
                        "current_mps_ms": current_mps_ms,
                        "package_cpu_ms": package_cpu_ms,
                        "package_mps_ms": "unsupported" if package_mps_ms is None else f"{package_mps_ms:.3f}",
                    },
                }
            )

    output_path.write_text(generate_markdown(results) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
