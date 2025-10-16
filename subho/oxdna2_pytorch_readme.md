# oxDNA2 PyTorch Implementation

This document explains the PyTorch implementation of the oxDNA2 coarse-grained force field for DNA simulations.

## Overview

The oxDNA2 model represents DNA nucleotides as rigid bodies with three interaction sites:
- **Backbone site**: Located at the phosphate group
- **Stacking site**: Located at the sugar
- **Base site**: Located at the nucleobase

Each nucleotide is defined by:
- Position (center of mass)
- Orientation (3x3 rotation matrix with columns v1, v2, v3)
- Base type (A, C, G, or T)

## Energy Terms

The oxDNA2 force field includes 8 different energy terms:

### 1. Backbone (FENE) - Bonded
**File Reference**: [DNAInteraction.cpp:409-460](src/Interactions/DNAInteraction.cpp#L409-L460)

A FENE (Finitely Extensible Nonlinear Elastic) potential connects bonded neighbors:

```
E_backbone = -(ε_FENE / 2) * log(1 - (r - r0)² / Δ²)
```

**Parameters**:
- `FENE_EPS = 2.0` - Energy scale
- `FENE_R0_OXDNA2 = 0.7564` - Equilibrium distance
- `FENE_DELTA = 0.25` - Maximum extension

### 2. Bonded Excluded Volume
**File Reference**: [DNAInteraction.cpp:462-522](src/Interactions/DNAInteraction.cpp#L462-L522)

Smoothed repulsive LJ potential between interaction sites of bonded neighbors:
- Base-Base
- Base-Backbone
- Backbone-Base

```
E_excl = 4ε[(σ/r)¹² - (σ/r)⁶]  for r < r*
E_excl = εb(r - rc)²            for r* ≤ r < rc
```

### 3. Stacking - Bonded
**File Reference**: [DNAInteraction.cpp:524-699](src/Interactions/DNAInteraction.cpp#L524-L699)

Base stacking between consecutive nucleotides on the same strand:

```
E_stack = f1(r) * f4(θ4) * f4(θ5) * f4(θ6) * f5(φ1) * f5(φ2)
```

**Components**:
- `f1(r)`: Modulated Morse potential (radial dependence)
- `f4(θ)`: Angular modulation functions
- `f5(φ)`: Dihedral-like modulation

**Angles**:
- θ4: angle between base normals (a3 · b3)
- θ5, θ6: angles with stacking direction
- φ1, φ2: twist angles around backbone

**oxDNA2 Parameters**:
- `STCK_BASE_EPS_OXDNA2 = 1.3523`
- `STCK_FACT_EPS_OXDNA2 = 2.6717`
- Energy depends on temperature: `ε = 1.3523 + 2.6717*T`

### 4. Hydrogen Bonding - Non-bonded
**File Reference**: [DNAInteraction.cpp:773-896](src/Interactions/DNAInteraction.cpp#L773-L896)

Watson-Crick base pairing (A-T and G-C):

```
E_HB = f1(r) * f4(θ1) * f4(θ2) * f4(θ3) * f4(θ4) * f4(θ7) * f4(θ8)
```

**Angles**:
- θ1: angle between bases (-a1 · b1)
- θ2, θ3: hydrogen bond directionality
- θ4: angle between base normals
- θ7, θ8: base plane alignment

**Parameters**:
- `HYDR_EPS_OXDNA2 = 1.0678` - Hydrogen bond strength

### 5. Cross-Stacking - Non-bonded
**File Reference**: [DNAInteraction.cpp:898-1015](src/Interactions/DNAInteraction.cpp#L898-L1015)

Stacking between bases on opposite strands:

```
E_cross = f2(r) * f4(θ1) * f4(θ2) * f4(θ3) * f4(θ4) * f4(θ7) * f4(θ8)
```

Uses modulated harmonic potential `f2(r)` instead of Morse.

### 6. Coaxial Stacking - Non-bonded
**File Reference**:
- [DNAInteraction.cpp:1017-1166](src/Interactions/DNAInteraction.cpp#L1017-L1166)
- [DNA2Interaction.cpp:206-298](src/Interactions/DNA2Interaction.cpp#L206-L298)

Stacking between bases on different helices (important for DNA junctions):

```
E_coax = f2(r) * f4(θ1) * f4(θ4) * f4(θ5) * f4(θ6) * f5²(φ3)
```

**oxDNA2 specific**:
- Modified `CXST_K_OXDNA2 = 58.5` (vs 46.0 in oxDNA)
- Special harmonic term for θ1 angle

### 7. Debye-Huckel Electrostatics - Non-bonded
**File Reference**: [DNA2Interaction.cpp:149-204](src/Interactions/DNA2Interaction.cpp#L149-L204)

Salt-dependent electrostatic repulsion between backbones:

```
E_DH = (q²/r) * exp(-κr)  for r < r_high
E_DH = B(r - r_c)²        for r_high ≤ r < r_c
```

**Parameters**:
- `κ = 1/λ` - Inverse Debye length
- `λ = 0.3616455 * √(T/0.1) / √(I)` - Debye length
- `I` - Salt concentration in Molars
- Half charges for terminus nucleotides (optional)

### 8. Non-bonded Excluded Volume
Similar to bonded excluded volume, but for all non-bonded pairs.

## Helper Functions

### f1(r) - Modulated Morse Potential
**File Reference**: [DNAInteraction.cpp:1210-1226](src/Interactions/DNAInteraction.cpp#L1210-L1226)

```python
if r_low < r < r_high:
    E = ε * (1 - exp(-a(r - r0)))² - shift
elif r_clow < r ≤ r_low:
    E = ε * B_low * (r - r_clow)²
elif r_high ≤ r < r_chigh:
    E = ε * B_high * (r - r_chigh)²
```

### f2(r) - Modulated Harmonic Potential
**File Reference**: [DNAInteraction.cpp:1246-1260](src/Interactions/DNAInteraction.cpp#L1246-L1260)

```python
if r_low < r < r_high:
    E = (k/2) * ((r - r0)² - (rc - r0)²)
elif r_clow < r ≤ r_low:
    E = k * B_low * (r - r_clow)²
elif r_high ≤ r < r_chigh:
    E = k * B_high * (r - r_chigh)²
```

### f4(θ) - Angular Modulation
**File Reference**: [DNAInteraction.cpp:1310-1326](src/Interactions/DNAInteraction.cpp#L1310-L1326)

```python
t = |θ - θ0|
if t < θ_s:
    f4 = 1 - a * t²
elif θ_s ≤ t < θ_c:
    f4 = b * (θ_c - t)²
else:
    f4 = 0
```

### f5(cos φ) - Dihedral Modulation
**File Reference**: [DNAInteraction.cpp:1379-1394](src/Interactions/DNAInteraction.cpp#L1379-L1394)

```python
if cos φ ≥ 0:
    f5 = 1
elif x_s ≤ cos φ < 0:
    f5 = 1 - a * (cos φ)²
elif x_c < cos φ < x_s:
    f5 = b * (x_c - cos φ)²
else:
    f5 = 0
```

## Model Constants

All model constants are defined in [src/model.h](src/model.h):

### Geometry
```cpp
POS_BACK = -0.4    // Backbone position
POS_STACK = 0.34   // Stacking site position
POS_BASE = 0.4     // Base position
```

### Key oxDNA2 Parameters (different from oxDNA)
```cpp
FENE_R0_OXDNA2 = 0.7564           // vs 0.7525 in oxDNA
STCK_BASE_EPS_OXDNA2 = 1.3523     // vs 1.3448 in oxDNA
STCK_FACT_EPS_OXDNA2 = 2.6717     // vs 2.6568 in oxDNA
HYDR_EPS_OXDNA2 = 1.0678          // vs 1.077 in oxDNA
CXST_K_OXDNA2 = 58.5              // vs 46.0 in oxDNA
CXST_THETA1_T0_OXDNA2 = π - 0.25  // vs π - 0.60 in oxDNA
```

## PyTorch Implementation

The implementation in `oxdna2_pytorch.py` provides:

### Main Class: `oxDNA2Energy`

```python
model = oxDNA2Energy(
    temperature=0.1,           # ~300K
    salt_concentration=0.5,    # 0.5 M
    use_average_seq=True,      # Sequence-averaged parameters
    grooving=True              # Major-minor groove model
)
```

### Input Format

The model expects nucleotide data as PyTorch tensors:

```python
# Positions: (N, 3) - center of mass coordinates
positions = torch.tensor([[x, y, z], ...])

# Orientations: (N, 3, 3) - rotation matrices
# Column 0 (v1): points from backbone to base
# Column 1 (v2): perpendicular to v1 and v3
# Column 2 (v3): stacking direction (helix axis)
orientations = torch.tensor([[[...], [...], [...]], ...])

# Base types: (N,) - 0=A, 1=C, 2=G, 3=T
base_types = torch.tensor([0, 3, 2, 1, ...])

# Bonding information: (N_pairs,) boolean
is_bonded = torch.tensor([True, True, False, ...])
```

### Computing Energies

```python
energies = model(
    positions_p, positions_q,
    orientations_p, orientations_q,
    is_bonded,
    base_type_p, base_type_q,
    is_terminus_p=None,  # Optional: terminus flags
    is_terminus_q=None
)

total_energy = energies.sum()
```

### Individual Energy Terms

You can also compute individual terms:

```python
# Stacking energy (bonded)
stacking = model.stacking_energy(pos_p, pos_q, ori_p, ori_q)

# Hydrogen bonding (non-bonded)
hbond = model.hydrogen_bonding_energy(pos_p, pos_q, ori_p, ori_q,
                                     base_p, base_q)

# Debye-Huckel
dh = model.debye_huckel_energy(pos_p, pos_q, ori_p, ori_q,
                               is_bonded, is_term_p, is_term_q)
```

## Differences from C++ Implementation

1. **No Meshes**: The C++ code uses lookup tables (meshes) for f4 functions. The PyTorch version computes them directly, which is simpler but potentially slower.

2. **Batched Operations**: PyTorch version operates on batches of particle pairs simultaneously.

3. **Automatic Differentiation**: PyTorch enables automatic computation of forces (gradients) with respect to positions and orientations.

4. **No Force Computation**: The C++ code includes force and torque calculations. The PyTorch version only computes energies (forces can be obtained via `energy.backward()`).

## Usage Example

```python
import torch
from oxdna2_pytorch import oxDNA2Energy

# Initialize model
model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

# Create sample data (2 nucleotides, bonded)
positions_p = torch.tensor([[0.0, 0.0, 0.0]])
positions_q = torch.tensor([[0.8, 0.0, 0.0]])

# Identity orientations
orientations_p = torch.eye(3).unsqueeze(0)
orientations_q = torch.eye(3).unsqueeze(0)

# A-T base pair
base_p = torch.tensor([0])  # A
base_q = torch.tensor([3])  # T
is_bonded = torch.tensor([True])

# Enable gradient computation
positions_p.requires_grad = True
positions_q.requires_grad = True

# Compute energy
energy = model(positions_p, positions_q,
               orientations_p, orientations_q,
               is_bonded, base_p, base_q)

# Compute forces (negative gradient)
energy.sum().backward()
force_p = -positions_p.grad
force_q = -positions_q.grad

print(f"Energy: {energy.item():.4f}")
print(f"Force on p: {force_p}")
print(f"Force on q: {force_q}")
```

## References

1. **oxDNA**: T. E. Ouldridge, A. A. Louis, and J. P. K. Doye, "Structural, mechanical, and thermodynamic properties of a coarse-grained DNA model", J. Chem. Phys. **134**, 085101 (2011)

2. **oxDNA2**: B. E. K. Snodin, F. Randisi, M. Mosayebi, et al., "Introducing improved structural properties and salt dependence into a coarse-grained model of DNA", J. Chem. Phys. **142**, 234901 (2015)

3. **Sequence Dependence**: P. Šulc, F. Romano, T. E. Ouldridge, et al., "Sequence-dependent thermodynamics of a coarse-grained DNA model", J. Chem. Phys. **137**, 135101 (2012)

## Files in C++ Implementation

- [src/model.h](src/model.h) - Model constants
- [src/Interactions/DNAInteraction.h](src/Interactions/DNAInteraction.h) - Base class header
- [src/Interactions/DNAInteraction.cpp](src/Interactions/DNAInteraction.cpp) - Base implementation
- [src/Interactions/DNA2Interaction.h](src/Interactions/DNA2Interaction.h) - oxDNA2 header
- [src/Interactions/DNA2Interaction.cpp](src/Interactions/DNA2Interaction.cpp) - oxDNA2 implementation
- [src/Particles/DNANucleotide.h](src/Particles/DNANucleotide.h) - Particle definition
- [src/Particles/DNANucleotide.cpp](src/Particles/DNANucleotide.cpp) - Particle implementation
