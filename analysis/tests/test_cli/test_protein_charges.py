"""
Tests for oxDNA_analysis_tools.protein_charges module.

Tests cover:
- reading protein beads from a DNANM/DNACT topology
- sequence-based and PQR-based charge assignment
- group assignment
- charge file format (as expected by the DNANM/DNACT parser)
- CLI argument parsing, main() entry point and error handling
"""
import sys
from unittest.mock import patch

import pytest

from oxDNA_analysis_tools.protein_charges import (
    ChargeError, assign_groups, cli_parser, main, pqr_charges, protein_charges,
    read_protein_topology, simple_charges,
)


# =============================================================================
# Fixtures
# =============================================================================

# Two protein strands (K E H | R D) followed by one DNA strand of two nucleotides.
# Header: <N> <N_strands> <N_DNA> <N_protein> <N_DNA_strands>
TOPOLOGY = """7 3 2 5 1
-1 K -1 1
-1 E 0 2
-1 H 1 -1
-2 R -1 4
-2 D 3 -1
1 A -1 6
1 T 5 -1
"""

PQR = """REMARK generated for the test
ATOM      1  N   LYS A   1      0.000   0.000   0.000  -0.30  1.82
ATOM      2  NZ  LYS A   1      1.000   0.000   0.000   1.30  1.82
ATOM      3  N   GLU A   2      2.000   0.000   0.000  -0.30  1.82
ATOM      4  OE1 GLU A   2      3.000   0.000   0.000  -0.70  1.82
ATOM      5  OE2 GLU A   2      3.500   0.000   0.000  -0.70  1.82
ATOM      6  ND1 HIP A   3      4.000   0.000   0.000   0.50  1.82
ATOM      7  NE2 HIP A   3      4.500   0.000   0.000   0.50  1.82
ATOM      8  NH1 ARG B   1      5.000   0.000   0.000   1.00  1.82
ATOM      9  OD1 ASP B   2      6.000   0.000   0.000  -1.00  1.82
TER
"""


@pytest.fixture
def topology(tmp_path):
    p = tmp_path / "test.top"
    p.write_text(TOPOLOGY)
    return str(p)


@pytest.fixture
def pqr(tmp_path):
    p = tmp_path / "test.pqr"
    p.write_text(PQR)
    return str(p)


def read_data_rows(path):
    return [l.split() for l in open(path) if l.strip() and not l.startswith('#')]


def read_meta(path):
    meta = {}
    for l in open(path):
        if l.startswith('#') and '=' in l:
            k, v = l[1:].split('=', 1)
            meta[k.strip()] = v.strip()
    return meta


# =============================================================================
# Topology
# =============================================================================

def test_read_protein_topology_skips_dna(topology):
    assert read_protein_topology(topology) == [(-1, 'K'), (-1, 'E'), (-1, 'H'), (-2, 'R'), (-2, 'D')]


def test_read_protein_topology_rejects_wrong_count(tmp_path):
    p = tmp_path / "bad.top"
    p.write_text(TOPOLOGY.replace("7 3 2 5 1", "7 3 2 4 1"))
    with pytest.raises(ChargeError):
        read_protein_topology(str(p))


def test_read_protein_topology_rejects_no_protein(tmp_path):
    p = tmp_path / "dna.top"
    p.write_text("2 1 2 0 1\n1 A -1 1\n1 T 0 -1\n")
    with pytest.raises(ChargeError):
        read_protein_topology(str(p))


# =============================================================================
# Charge models
# =============================================================================

def test_simple_charges(topology):
    residues = read_protein_topology(topology)
    assert simple_charges(residues) == [1.0, -1.0, 0.0, 1.0, -1.0]
    assert simple_charges(residues, his_charge=0.5)[2] == 0.5


def test_simple_charges_termini_are_per_strand(topology):
    residues = read_protein_topology(topology)
    # +1 on the first and -1 on the last bead of each strand
    assert simple_charges(residues, termini=True) == [2.0, -1.0, -1.0, 2.0, -2.0]


def test_pqr_charges_sum_per_residue(topology, pqr):
    residues = read_protein_topology(topology)
    charges = pqr_charges(residues, pqr)
    # LYS: -0.30 + 1.30, GLU: -0.30 - 0.70 - 0.70, HIP: 0.50 + 0.50, ARG: 1.00, ASP: -1.00
    assert charges == pytest.approx([1.0, -1.7, 1.0, 1.0, -1.0], abs=1e-9)


def test_pqr_charges_reject_residue_mismatch(tmp_path, topology):
    bad = tmp_path / "bad.pqr"
    bad.write_text(PQR.replace("ARG B   1", "LYS B   1"))
    with pytest.raises(ChargeError):
        pqr_charges(read_protein_topology(topology), str(bad))


def test_pqr_charges_reject_wrong_length(tmp_path, topology):
    short = tmp_path / "short.pqr"
    short.write_text("\n".join(PQR.splitlines()[:-3]) + "\n")
    with pytest.raises(ChargeError):
        pqr_charges(read_protein_topology(topology), str(short))


# =============================================================================
# Groups
# =============================================================================

def test_assign_groups_modes(topology):
    residues = read_protein_topology(topology)
    assert assign_groups(residues, 'strand') == [0, 0, 0, 1, 1]
    assert assign_groups(residues, 'single') == [0, 0, 0, 0, 0]
    assert assign_groups(residues, 'residue') == [0, 1, 2, 3, 4]


def test_assign_groups_file_overrides_and_is_renumbered(tmp_path, topology):
    residues = read_protein_topology(topology)
    gf = tmp_path / "groups.txt"
    gf.write_text("7\n7\n3\n3\n3\n")
    assert assign_groups(residues, 'strand', str(gf)) == [0, 0, 1, 1, 1]


def test_assign_groups_file_wrong_length(tmp_path, topology):
    gf = tmp_path / "groups.txt"
    gf.write_text("0\n0\n")
    with pytest.raises(ChargeError):
        assign_groups(read_protein_topology(topology), 'strand', str(gf))


# =============================================================================
# File format
# =============================================================================

def test_charge_file_format(tmp_path, topology):
    out = tmp_path / "charges.txt"
    charges, groups = protein_charges(topology, str(out))

    # the parser requires the versioned header to come first
    assert open(out).readline().strip() == "# DNANM_ES_CHARGES 1"

    meta = read_meta(out)
    assert meta["index_space"] == "protein_local"
    assert meta["index_base"] == "0"
    assert meta["n_protein_beads"] == "5"
    assert meta["n_groups"] == "2"

    rows = read_data_rows(out)
    assert [r[:3] for r in rows] == [["0", "-1", "K"], ["1", "-1", "E"], ["2", "-1", "H"],
                                     ["3", "-2", "R"], ["4", "-2", "D"]]
    assert [float(r[3]) for r in rows] == charges
    assert [int(r[4]) for r in rows] == groups
    # no duplicated metadata keys (the parser rejects them)
    keys = [l[1:].split('=', 1)[0].strip() for l in open(out) if l.startswith('#') and '=' in l]
    assert len(keys) == len(set(keys))


def test_charge_file_with_pqr(tmp_path, topology, pqr):
    out = tmp_path / "charges.txt"
    protein_charges(topology, str(out), pqr=pqr)
    assert "PQR" in read_meta(out)["charge_model"]
    assert [float(r[3]) for r in read_data_rows(out)] == pytest.approx([1.0, -1.7, 1.0, 1.0, -1.0], abs=1e-5)


# =============================================================================
# CLI
# =============================================================================

def test_cli_parser_defaults():
    args = cli_parser().parse_args(["in.top", "out.txt"])
    assert args.pqr is None
    assert args.his_charge == 0.0
    assert args.termini is False
    assert args.group_by == "strand"
    assert args.group_file is None


def test_main_writes_file(tmp_path, topology, capsys):
    out = tmp_path / "charges.txt"
    with patch.object(sys, 'argv', ["protein_charges.py", topology, str(out), "--his-charge", "0.1"]):
        main()
    assert out.exists()
    assert "Net protein charge" in capsys.readouterr().out
    assert float(read_data_rows(out)[2][3]) == pytest.approx(0.1)


def test_main_reports_errors(tmp_path, capsys):
    with patch.object(sys, 'argv', ["protein_charges.py", str(tmp_path / "missing.top"), str(tmp_path / "o.txt")]):
        with pytest.raises(SystemExit) as e:
            main()
    assert e.value.code == 1
    assert "ERROR" in capsys.readouterr().err
