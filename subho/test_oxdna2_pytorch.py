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
