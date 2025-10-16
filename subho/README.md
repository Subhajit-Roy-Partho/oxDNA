# oxDNA2 PyTorch Implementation

A complete PyTorch implementation of the oxDNA2 coarse-grained force field for DNA simulations, converted from the original C++ implementation.

## Contents

- `oxdna2_pytorch.py` - Main PyTorch implementation
- `oxdna2_pytorch_readme.md` - Detailed documentation of the model
- `test_oxdna2_pytorch.py` - Test suite with examples
- `README.md` - This file

## What is oxDNA2?

oxDNA2 is a coarse-grained model for simulating DNA at the nucleotide level. Each nucleotide is represented as a rigid body with three interaction sites (backbone, stacking, and base), capturing the essential physics of DNA including:

- Base stacking
- Watson-Crick hydrogen bonding
- Electrostatic repulsion
- Structural constraints
- Salt-dependent interactions

## Key Features of this Implementation

1. **Complete Energy Terms**: All 8 energy terms from oxDNA2 are implemented:
   - FENE backbone potential
   - Bonded excluded volume
   - Stacking (bonded)
   - Hydrogen bonding (non-bonded, Watson-Crick)
   - Cross-stacking (non-bonded)
   - Coaxial stacking (non-bonded)
   - Debye-Huckel electrostatics
   - Non-bonded excluded volume

2. **Automatic Differentiation**: Forces can be computed automatically via PyTorch's autograd

3. **Batch Processing**: Efficiently compute energies for multiple particle pairs simultaneously

4. **GPU Compatible**: Can be run on GPU for faster computation

5. **Faithful Translation**: Directly translated from the C++ implementation with all parameters and functional forms preserved

## Quick Start

### Installation

```bash
# Required packages
pip install torch numpy
```

### Basic Usage

```python
import torch
from oxdna2_pytorch import oxDNA2Energy

# Initialize the model
model = oxDNA2Energy(
    temperature=0.1,           # Simulation units (~300K)
    salt_concentration=0.5     # Molars
)

# Create nucleotide data
positions_p = torch.tensor([[0.0, 0.0, 0.0]])
positions_q = torch.tensor([[0.8, 0.0, 0.0]])

# Orientations (3x3 rotation matrices)
orientations_p = torch.eye(3).unsqueeze(0)
orientations_q = torch.eye(3).unsqueeze(0)

# Base types: 0=A, 1=C, 2=G, 3=T
base_type_p = torch.tensor([0])  # Adenine
base_type_q = torch.tensor([3])  # Thymine

# Is this pair bonded?
is_bonded = torch.tensor([True])

# Compute energy
energy = model(
    positions_p, positions_q,
    orientations_p, orientations_q,
    is_bonded,
    base_type_p, base_type_q
)

print(f"Energy: {energy.item()}")
```

### Computing Forces

```python
# Enable gradient computation
positions_p.requires_grad = True
positions_q.requires_grad = True

# Compute energy
energy = model(...)

# Compute forces (negative gradient)
energy.sum().backward()
force_p = -positions_p.grad
force_q = -positions_q.grad
```

## Running Tests

```bash
cd subho
python test_oxdna2_pytorch.py
```

The test suite includes:
1. Bonded pair at equilibrium
2. Hydrogen bonding between complementary bases
3. Non-complementary bases (no H-bonding)
4. Debye-Huckel vs. distance
5. Force computation via autograd
6. Batch processing
7. Temperature dependence
8. Salt concentration dependence

## Understanding the Model

### Nucleotide Representation

Each nucleotide is a rigid body with:
- **Position**: Center of mass (COM)
- **Orientation**: 3x3 rotation matrix where:
  - Column 0 (v1): Points from backbone to base
  - Column 1 (v2): Perpendicular to v1 and v3
  - Column 2 (v3): Stacking direction (helix axis)

### Interaction Sites

Three sites are defined relative to the COM:
- **Backbone**: position = COM + v1 × (-0.4)
- **Stacking**: position = COM + v1 × (0.34)
- **Base**: position = COM + v1 × (0.4)

### Energy Terms

#### 1. Backbone (FENE)
Connects bonded neighbors with a finitely extensible spring:
```
E = -(ε/2) log(1 - (r-r₀)²/Δ²)
```

#### 2. Stacking
Stabilizes base stacking between consecutive nucleotides:
```
E = f₁(r) × f₄(θ₄) × f₄(θ₅) × f₄(θ₆) × f₅(φ₁) × f₅(φ₂)
```

#### 3. Hydrogen Bonding
Watson-Crick base pairing (A-T, G-C):
```
E = f₁(r) × f₄(θ₁) × f₄(θ₂) × f₄(θ₃) × f₄(θ₄) × f₄(θ₇) × f₄(θ₈)
```

#### 4. Debye-Huckel
Salt-dependent electrostatic repulsion:
```
E = (q²/r) exp(-κr)
```
where κ = 1/λ, and λ is the Debye screening length.

See `oxdna2_pytorch_readme.md` for complete details on all energy terms.

## Model Parameters

### Temperature
- Simulation units: 0.1 ≈ 300K
- Room temperature: T = 0.1
- Melting studies: T = 0.08 to 0.12

### Salt Concentration
- Physiological: ~0.15 M (150 mM)
- Common simulation: 0.5 M
- Range: 0.1 M to 2.0 M

### oxDNA2 vs oxDNA

Key differences:
- Stronger stacking (1.3523 vs 1.3448)
- Modified hydrogen bonding (1.0678 vs 1.077)
- Enhanced coaxial stacking (k=58.5 vs 46.0)
- Salt-dependent Debye-Huckel term
- Modified FENE equilibrium (0.7564 vs 0.7525)

## File Structure

```
subho/
├── README.md                      # This file
├── oxdna2_pytorch.py             # Main implementation (~900 lines)
├── oxdna2_pytorch_readme.md      # Detailed documentation
└── test_oxdna2_pytorch.py        # Test suite
```

## Original C++ Implementation

This PyTorch version was translated from the oxDNA C++ code:

```
oxDNA/src/
├── model.h                       # Model constants
├── Interactions/
│   ├── DNAInteraction.h         # Base class
│   ├── DNAInteraction.cpp       # Base implementation
│   ├── DNA2Interaction.h        # oxDNA2 header
│   └── DNA2Interaction.cpp      # oxDNA2 implementation
└── Particles/
    ├── DNANucleotide.h          # Particle definition
    └── DNANucleotide.cpp        # Particle implementation
```

## Validation

To validate against the C++ implementation:

1. Load the same configuration in both implementations
2. Compute energies for identical particle pairs
3. Compare results (should match to floating-point precision)

Note: The PyTorch version computes f4 functions directly rather than using lookup tables (meshes), which may introduce small numerical differences.

## Use Cases

1. **Machine Learning**: Train neural networks on DNA structures
2. **Energy Evaluation**: Quickly evaluate energies for large datasets
3. **Force Computation**: Automatic differentiation for forces
4. **Analysis**: Batch processing of trajectory data
5. **Method Development**: Rapid prototyping of new algorithms

## Performance Notes

- **CPU vs GPU**: For small systems, CPU may be faster due to overhead. GPU is beneficial for large batches.
- **Batch Size**: Optimal batch size depends on hardware. Typical: 100-1000 pairs.
- **Memory**: Each particle pair requires storage for positions, orientations, and intermediate calculations.

## Limitations

1. **No Sequence Dependence**: Currently uses sequence-averaged parameters. Sequence-dependent parameters can be added by modifying epsilon values in `__init__`.

2. **No Major-Minor Groove**: Simplified groove model (can be enabled by setting `grooving=True` and adjusting backbone positions).

3. **No Meshes**: f4 functions computed directly rather than via lookup tables. This is simpler but potentially slower.

4. **Pairwise Only**: Computes energies for pairs. For full system energy, you need to sum over all relevant pairs.

## Future Enhancements

Possible extensions:
- [ ] Sequence-dependent parameters
- [ ] Major-minor groove model
- [ ] Mesh-based f4 functions for speed
- [ ] Full system energy calculator
- [ ] Integration with molecular dynamics
- [ ] Visualization tools
- [ ] Trajectory analysis utilities

## References

1. **oxDNA**: T. E. Ouldridge, A. A. Louis, and J. P. K. Doye, "Structural, mechanical, and thermodynamic properties of a coarse-grained DNA model", *J. Chem. Phys.* **134**, 085101 (2011)

2. **oxDNA2**: B. E. K. Snodin, F. Randisi, M. Mosayebi, et al., "Introducing improved structural properties and salt dependence into a coarse-grained model of DNA", *J. Chem. Phys.* **142**, 234901 (2015)

3. **Sequence Dependence**: P. Šulc, F. Romano, T. E. Ouldridge, et al., "Sequence-dependent thermodynamics of a coarse-grained DNA model", *J. Chem. Phys.* **137**, 135101 (2012)

## Citation

If you use this implementation, please cite the original oxDNA papers and acknowledge this PyTorch implementation.

## License

This implementation follows the same license as the original oxDNA code.

## Contact

For questions about this implementation, please refer to the oxDNA documentation at:
https://dna.physics.ox.ac.uk/

Original C++ code:
https://github.com/lorenzo-rovigatti/oxDNA

## Acknowledgments

This PyTorch implementation was created by carefully translating the C++ oxDNA2 code, preserving all physical parameters and functional forms. Thanks to the oxDNA developers for creating and maintaining the original implementation.
