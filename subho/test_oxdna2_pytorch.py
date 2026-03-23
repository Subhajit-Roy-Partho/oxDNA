"""
Test script for oxDNA2 PyTorch implementation.

This script demonstrates how to use the oxDNA2 PyTorch energy calculator
with simple examples.
"""

import torch
import numpy as np
from oxdna2_pytorch import oxDNA2Energy


def test_basic_bonded_pair():
    """Test energy calculation for a simple bonded pair."""
    print("=" * 60)
    print("Test 1: Bonded pair at equilibrium distance")
    print("=" * 60)

    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    # Two bonded nucleotides at approximately equilibrium distance
    positions_p = torch.tensor([[0.0, 0.0, 0.0]])
    positions_q = torch.tensor([[0.7564, 0.0, 0.0]])  # FENE_R0_OXDNA2

    # Simple orientations (identity)
    orientations_p = torch.eye(3).unsqueeze(0)
    orientations_q = torch.eye(3).unsqueeze(0)

    # Any base types (doesn't matter for bonded interactions)
    base_p = torch.tensor([0])  # A
    base_q = torch.tensor([1])  # C
    is_bonded = torch.tensor([True])

    energy = model(positions_p, positions_q,
                   orientations_p, orientations_q,
                   is_bonded, base_p, base_q)

    print(f"Total energy: {energy.item():.6f}")
    print(f"(Should be negative, dominated by stacking)")
    print()


def test_hydrogen_bonding():
    """Test hydrogen bonding between complementary bases."""
    print("=" * 60)
    print("Test 2: Hydrogen bonding (A-T Watson-Crick pair)")
    print("=" * 60)

    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    # Two non-bonded nucleotides facing each other
    # Positioned for hydrogen bonding
    positions_p = torch.tensor([[0.0, 0.0, 0.0]])
    positions_q = torch.tensor([[0.0, 0.0, 1.2]])  # Typical HB distance

    # Orientations: bases facing each other
    # p's base points in +z direction, q's in -z direction
    ori_p = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]])
    ori_q = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]]])

    # Complementary base pair (A-T)
    base_p = torch.tensor([0])  # A
    base_q = torch.tensor([3])  # T
    is_bonded = torch.tensor([False])

    energy = model(positions_p, positions_q,
                   ori_p, ori_q,
                   is_bonded, base_p, base_q)

    print(f"Total energy: {energy.item():.6f}")
    print(f"(Should be negative if geometry is favorable)")
    print()


def test_non_complementary_bases():
    """Test that non-complementary bases don't form hydrogen bonds."""
    print("=" * 60)
    print("Test 3: Non-complementary bases (A-A, should not H-bond)")
    print("=" * 60)

    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    positions_p = torch.tensor([[0.0, 0.0, 0.0]])
    positions_q = torch.tensor([[0.0, 0.0, 1.2]])

    ori_p = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]])
    ori_q = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]]])

    # Non-complementary bases (A-A)
    base_p = torch.tensor([0])  # A
    base_q = torch.tensor([0])  # A (not complementary)
    is_bonded = torch.tensor([False])

    energy = model(positions_p, positions_q,
                   ori_p, ori_q,
                   is_bonded, base_p, base_q)

    print(f"Total energy: {energy.item():.6f}")
    print(f"(Should have no H-bonding contribution, mainly excluded volume)")
    print()


def test_debye_huckel_vs_distance():
    """Test Debye-Huckel energy as a function of distance."""
    print("=" * 60)
    print("Test 4: Debye-Huckel electrostatics vs. distance")
    print("=" * 60)

    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    # Test at several distances
    distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"{'Distance':<12} {'DH Energy':<15}")
    print("-" * 30)

    for d in distances:
        positions_p = torch.tensor([[0.0, 0.0, 0.0]])
        positions_q = torch.tensor([[d, 0.0, 0.0]])

        ori_p = torch.eye(3).unsqueeze(0)
        ori_q = torch.eye(3).unsqueeze(0)

        base_p = torch.tensor([0])
        base_q = torch.tensor([0])
        is_bonded = torch.tensor([False])

        # Compute only DH energy
        dh_energy = model.debye_huckel_energy(
            positions_p, positions_q, ori_p, ori_q,
            is_bonded, torch.tensor([False]), torch.tensor([False])
        )

        print(f"{d:<12.2f} {dh_energy.item():<15.6f}")

    print()


def test_gradient_computation():
    """Test automatic differentiation for force computation."""
    print("=" * 60)
    print("Test 5: Force computation via automatic differentiation")
    print("=" * 60)

    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    # Create positions with gradient tracking
    positions_p = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    positions_q = torch.tensor([[0.8, 0.0, 0.0]], requires_grad=True)

    ori_p = torch.eye(3).unsqueeze(0)
    ori_q = torch.eye(3).unsqueeze(0)

    base_p = torch.tensor([0])
    base_q = torch.tensor([3])
    is_bonded = torch.tensor([True])

    # Compute energy
    energy = model(positions_p, positions_q,
                   ori_p, ori_q,
                   is_bonded, base_p, base_q)

    # Compute forces (negative gradient)
    energy.sum().backward()

    force_p = -positions_p.grad
    force_q = -positions_q.grad

    print(f"Energy: {energy.item():.6f}")
    print(f"Force on particle p: [{force_p[0,0].item():.6f}, "
          f"{force_p[0,1].item():.6f}, {force_p[0,2].item():.6f}]")
    print(f"Force on particle q: [{force_q[0,0].item():.6f}, "
          f"{force_q[0,1].item():.6f}, {force_q[0,2].item():.6f}]")
    print(f"Force balance (should be zero): "
          f"{(force_p + force_q).norm().item():.8f}")
    print()


def test_batch_computation():
    """Test batch computation of multiple particle pairs."""
    print("=" * 60)
    print("Test 6: Batch computation (multiple pairs)")
    print("=" * 60)

    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    # Create batch of 5 pairs
    batch_size = 5
    positions_p = torch.randn(batch_size, 3)
    positions_q = positions_p + torch.tensor([0.8, 0.0, 0.0])

    ori_p = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    ori_q = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)

    # Mix of bonded and non-bonded pairs
    is_bonded = torch.tensor([True, True, False, False, False])

    # Various base types
    base_p = torch.tensor([0, 1, 2, 3, 0])  # A, C, G, T, A
    base_q = torch.tensor([3, 2, 1, 0, 2])  # T, G, C, A, G

    energies = model(positions_p, positions_q,
                     ori_p, ori_q,
                     is_bonded, base_p, base_q)

    print(f"Individual pair energies:")
    for i, e in enumerate(energies):
        pair_type = "bonded" if is_bonded[i] else "non-bonded"
        print(f"  Pair {i+1} ({pair_type}): {e.item():.6f}")

    print(f"\nTotal energy: {energies.sum().item():.6f}")
    print()


def test_temperature_dependence():
    """Test temperature dependence of stacking energy."""
    print("=" * 60)
    print("Test 7: Temperature dependence")
    print("=" * 60)

    temperatures = [0.08, 0.10, 0.12]  # ~240K, 300K, 360K

    print(f"{'T (sim units)':<15} {'~T (K)':<12} {'Energy':<15}")
    print("-" * 45)

    for T in temperatures:
        model = oxDNA2Energy(temperature=T, salt_concentration=0.5)

        positions_p = torch.tensor([[0.0, 0.0, 0.0]])
        positions_q = torch.tensor([[0.75, 0.0, 0.0]])

        ori_p = torch.eye(3).unsqueeze(0)
        ori_q = torch.eye(3).unsqueeze(0)

        base_p = torch.tensor([0])
        base_q = torch.tensor([1])
        is_bonded = torch.tensor([True])

        energy = model(positions_p, positions_q,
                       ori_p, ori_q,
                       is_bonded, base_p, base_q)

        T_kelvin = T * 3000
        print(f"{T:<15.2f} {T_kelvin:<12.0f} {energy.item():<15.6f}")

    print()


def test_salt_dependence():
    """Test salt concentration dependence of Debye-Huckel."""
    print("=" * 60)
    print("Test 8: Salt concentration dependence")
    print("=" * 60)

    salt_concs = [0.1, 0.5, 1.0, 2.0]  # Molars

    print(f"{'Salt (M)':<15} {'DH Energy':<15} {'Debye length':<15}")
    print("-" * 50)

    for salt in salt_concs:
        model = oxDNA2Energy(temperature=0.1, salt_concentration=salt)

        positions_p = torch.tensor([[0.0, 0.0, 0.0]])
        positions_q = torch.tensor([[1.0, 0.0, 0.0]])

        ori_p = torch.eye(3).unsqueeze(0)
        ori_q = torch.eye(3).unsqueeze(0)

        base_p = torch.tensor([0])
        base_q = torch.tensor([0])
        is_bonded = torch.tensor([False])

        dh_energy = model.debye_huckel_energy(
            positions_p, positions_q, ori_p, ori_q,
            is_bonded, torch.tensor([False]), torch.tensor([False])
        )

        # Debye length
        lambda_dh = 0.3616455 * np.sqrt(0.1 / 0.1) / np.sqrt(salt)

        print(f"{salt:<15.2f} {dh_energy.item():<15.6f} {lambda_dh:<15.6f}")

    print()


def test_sequence_parameter_buffers_follow_model_dtype():
    """Sequence-parameter tables should follow module dtype conversions."""
    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5).double()

    assert model.stck_eps_matrix.dtype == torch.float64
    assert model.stck_shift_matrix.dtype == torch.float64
    assert model.hydr_eps_matrix.dtype == torch.float64
    assert model.hydr_shift_matrix.dtype == torch.float64


def test_forward_matches_system_path_when_bond_order_is_provided():
    """`forward()` should match `compute_system_energies()` for reversed bonded order."""
    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5, use_average_seq=False)

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.7564, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    orientations = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    base_types = torch.tensor([0, 1], dtype=torch.long)  # A, C

    # Reverse the strand order so pair (0, 1) is bonded but 1 is the 3' neighbor of 0 is false.
    n3_neighbors = torch.tensor([-1, 0], dtype=torch.long)
    n5_neighbors = torch.tensor([1, -1], dtype=torch.long)

    system = model.compute_system_energies(
        positions, orientations, base_types, n3_neighbors, n5_neighbors
    )

    pair_indices = system["pair_indices"]
    i_idx = pair_indices[0]
    j_idx = pair_indices[1]
    is_bonded = (n3_neighbors[i_idx] == j_idx) | (n5_neighbors[i_idx] == j_idx)
    q_is_n3_of_p = n3_neighbors[i_idx] == j_idx

    pair_total = model(
        positions[i_idx],
        positions[j_idx],
        orientations[i_idx],
        orientations[j_idx],
        is_bonded,
        base_types[i_idx],
        base_types[j_idx],
        q_is_n3_of_p=q_is_n3_of_p,
    )

    assert torch.allclose(pair_total, system["pair_total"], atol=1e-6, rtol=1e-6)


def test_backbone_pair_energy_matches_double_precision_geometry():
    """Bonded FENE should use float64 backbone geometry when available."""
    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    positions_p = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    positions_q = torch.tensor([[0.7564, 0.01, -0.02]], dtype=torch.float32)

    orientations_p = torch.tensor(
        [[[0.99, 0.0, 0.1], [0.0, 1.0, 0.0], [-0.1, 0.0, 0.99]]],
        dtype=torch.float32,
    )
    orientations_q = torch.tensor(
        [[[0.98, 0.02, -0.18], [-0.02, 1.0, 0.01], [0.18, -0.01, 0.98]]],
        dtype=torch.float32,
    )

    energy = model.backbone_pair_energy(positions_p, positions_q, orientations_p, orientations_q)

    positions_p64 = positions_p.double()
    positions_q64 = positions_q.double()
    orientations_p64 = orientations_p.double()
    orientations_q64 = orientations_q.double()
    back_p64 = model._backbone_site(positions_p64, orientations_p64)
    back_q64 = model._backbone_site(positions_q64, orientations_q64)
    ref_energy = model.backbone_energy(torch.norm(back_q64 - back_p64, dim=1)).to(torch.float32)

    assert torch.allclose(energy, ref_energy, atol=1e-7, rtol=1e-7)


def test_compute_dtype_option_controls_output_dtype():
    """Explicit compute_dtype should control the dtype of the system path outputs."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.7564, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    orientations = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    base_types = torch.tensor([0, 1], dtype=torch.long)
    n3_neighbors = torch.tensor([1, -1], dtype=torch.long)
    n5_neighbors = torch.tensor([-1, 0], dtype=torch.long)

    model64 = oxDNA2Energy(temperature=0.1, salt_concentration=0.5, compute_dtype="float64")
    result64 = model64.compute_system_energies(positions, orientations, base_types, n3_neighbors, n5_neighbors)
    assert result64["pair_total"].dtype == torch.float64

    model32 = oxDNA2Energy(temperature=0.1, salt_concentration=0.5, compute_dtype="float32")
    result32 = model32.compute_system_energies(positions.double(), orientations.double(), base_types, n3_neighbors, n5_neighbors)
    assert result32["pair_total"].dtype == torch.float32

    model16 = oxDNA2Energy(temperature=0.1, salt_concentration=0.5, compute_dtype="float16")
    result16 = model16.compute_system_energies(positions, orientations, base_types, n3_neighbors, n5_neighbors)
    assert result16["pair_total"].dtype == torch.float16

    model8 = oxDNA2Energy(temperature=0.1, salt_concentration=0.5, compute_dtype="float8_e4m3fn")
    result8 = model8.compute_system_energies(positions, orientations, base_types, n3_neighbors, n5_neighbors)
    assert result8["pair_total"].dtype == torch.float16


def test_f1_sequence_dependent_masking_handles_partial_regions():
    """Vector-valued eps/shift must be masked consistently inside _f1."""
    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)
    r = torch.tensor([0.20, 0.40, 0.80], dtype=torch.float32)
    eps = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    shift = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    out = model._f1(r, eps, shift, 6.0, 0.4, 0.32, 0.75, 0.23239, 0.956, -68.1857, -3.12992)

    assert out.shape == r.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("oxDNA2 PyTorch Implementation - Test Suite")
    print("=" * 60 + "\n")

    # Run all tests
    test_basic_bonded_pair()
    test_hydrogen_bonding()
    test_non_complementary_bases()
    test_debye_huckel_vs_distance()
    test_gradient_computation()
    test_batch_computation()
    test_temperature_dependence()
    test_salt_dependence()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
