# oxDNA Energy Analysis Tools - Complete Index

This directory contains a complete set of tools and documentation for analyzing energies in oxDNA simulations.

## Quick Start

**Want to visualize energies in oxView?**
```bash
python -m oxDNA_analysis_tools.output_bonds input.dat trajectory.dat -v output
```

**Want correct per-nucleotide energies for analysis?**
```bash
cd subho
python output_per_nucleotide_energy.py input.dat trajectory.dat -v output
```

**Want to use PyTorch for custom analysis?**
```python
from oxdna2_pytorch import oxDNA2Energy
model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)
# See README.md for examples
```

---

## Files Overview

### 📘 Documentation (Start Here!)

| File | Description | Read Time |
|------|-------------|-----------|
| [**ENERGY_ANALYSIS_GUIDE.md**](ENERGY_ANALYSIS_GUIDE.md) | **Start here!** Complete guide to energy analysis | 20 min |
| [README.md](README.md) | Overview of PyTorch implementation | 10 min |
| [oxdna2_pytorch_readme.md](oxdna2_pytorch_readme.md) | Technical details of oxDNA2 model | 15 min |
| [output_bonds_explanation.md](output_bonds_explanation.md) | How `output_bonds.py` works | 10 min |

### 🔧 Tools

| File | Purpose | Lines |
|------|---------|-------|
| [**oxdna2_pytorch.py**](oxdna2_pytorch.py) | PyTorch implementation of oxDNA2 force field | 881 |
| [**output_per_nucleotide_energy.py**](output_per_nucleotide_energy.py) | Modified energy calculator (correct accounting) | 312 |
| [**test_oxdna2_pytorch.py**](test_oxdna2_pytorch.py) | Test suite with 8 examples | 311 |

### 📊 Total

- **7 files**
- **~3000 lines** of code and documentation
- Covers energy calculation, analysis, and visualization

---

## What's Included

### 1. PyTorch Implementation of oxDNA2

**File**: `oxdna2_pytorch.py`

A complete PyTorch reimplementation of the oxDNA2 force field with all 8 energy terms:

✅ FENE backbone
✅ Bonded excluded volume
✅ Stacking
✅ Hydrogen bonding
✅ Cross-stacking
✅ Coaxial stacking (with oxDNA2 modifications)
✅ Debye-Huckel electrostatics
✅ Non-bonded excluded volume

**Features**:
- Batch processing
- Automatic differentiation for forces
- GPU support
- Faithful to C++ implementation

**Example**:
```python
from oxdna2_pytorch import oxDNA2Energy

model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)
energy = model(positions_p, positions_q, orientations_p, orientations_q,
               is_bonded, base_type_p, base_type_q)
```

### 2. Per-Nucleotide Energy Calculator

**File**: `output_per_nucleotide_energy.py`

Modified version of oxDNA's `output_bonds.py` that provides **correct per-nucleotide energies**:

**Key difference**:
- Original: Adds full pairwise energy to both particles (double counting)
- Modified: Splits pairwise energy equally (no double counting)

**Usage**:
```bash
# Correct accounting (recommended for analysis)
python output_per_nucleotide_energy.py input.dat traj.dat -v output

# Original behavior (for comparison with output_bonds.py)
python output_per_nucleotide_energy.py input.dat traj.dat -v output --no-split

# With units and parallel processing
python output_per_nucleotide_energy.py input.dat traj.dat -v output -u pNnm -p 4
```

**Output**:
- Separate JSON files for each energy term
- Compatible with oxView
- Correct total system energy
- Energy breakdown statistics

### 3. Comprehensive Test Suite

**File**: `test_oxdna2_pytorch.py`

8 tests demonstrating all features:

1. ✅ Bonded pair at equilibrium
2. ✅ Hydrogen bonding (A-T pairing)
3. ✅ Non-complementary bases (no H-bond)
4. ✅ Debye-Huckel vs distance
5. ✅ Force computation via autograd
6. ✅ Batch processing
7. ✅ Temperature dependence
8. ✅ Salt concentration dependence

**Run all tests**:
```bash
cd subho
python test_oxdna2_pytorch.py
```

### 4. Complete Documentation

#### ENERGY_ANALYSIS_GUIDE.md
**The master guide** covering:
- Understanding energy accounting (split vs double-count)
- When to use each tool
- How to analyze energies
- Visualization in oxView
- Common analysis tasks
- Troubleshooting

#### README.md
Quick start guide for the PyTorch implementation:
- Installation
- Basic usage
- Examples
- Performance notes

#### oxdna2_pytorch_readme.md
Technical reference:
- All energy terms with equations
- Parameter values
- Model constants
- File references to C++ code
- Differences between oxDNA and oxDNA2

#### output_bonds_explanation.md
Deep dive into the original tool:
- How `output_bonds.py` works
- Understanding the C++ observable
- Energy accounting explained
- How to modify for custom needs

---

## Workflows

### Workflow 1: Quick Visualization

```bash
# Generate energy overlays for oxView
python -m oxDNA_analysis_tools.output_bonds input.dat trajectory.dat -v mystructure

# Load in oxView
# 1. Go to https://sulcgroup.github.io/oxdna-viewer/
# 2. Load configuration and topology
# 3. Overlay → Load file → Select mystructure_STCK.json
```

### Workflow 2: Quantitative Analysis

```bash
# Compute per-nucleotide energies (correct accounting)
cd subho
python output_per_nucleotide_energy.py input.dat trajectory.dat -v analysis

# Analyze in Python
python
>>> import numpy as np
>>> import json
>>> with open('analysis_total.json') as f:
...     data = json.load(f)
>>> energies = np.array(list(data.values())[0])
>>> print(f"Mean: {energies.mean():.3f}, Std: {energies.std():.3f}")
>>> unstable = np.where(energies > energies.mean() + 2*energies.std())[0]
>>> print(f"Unstable nucleotides: {unstable}")
```

### Workflow 3: Machine Learning

```python
import torch
from oxdna2_pytorch import oxDNA2Energy

# Initialize model
model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

# Load trajectory data
positions_p, positions_q = load_pairs_from_trajectory()
orientations_p, orientations_q = load_orientations()
is_bonded, base_types = load_topology()

# Compute energies for all pairs
energies = model(positions_p, positions_q,
                 orientations_p, orientations_q,
                 is_bonded, base_types[:, 0], base_types[:, 1])

# Use in ML pipeline
dataset = TensorDataset(positions_p, positions_q, energies)
loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    # Train your model
    ...
```

### Workflow 4: Custom Energy Analysis

```python
from oxdna2_pytorch import oxDNA2Energy

model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

# Compute individual energy terms
stacking = model.stacking_energy(pos_p, pos_q, ori_p, ori_q)
hbonding = model.hydrogen_bonding_energy(pos_p, pos_q, ori_p, ori_q,
                                         base_p, base_q)

# Compare different salt concentrations
energies_low_salt = []
energies_high_salt = []

for salt in [0.1, 0.5, 1.0, 2.0]:
    model_salt = oxDNA2Energy(temperature=0.1, salt_concentration=salt)
    E = model_salt(...)
    energies_by_salt.append(E)

# Analyze salt dependence
...
```

---

## Key Concepts

### Energy Accounting

**Pairwise Energy**: E(i,j)
- Fundamental quantity computed by the force field
- Represents interaction between nucleotides i and j

**Per-Nucleotide Energy**: E[i]
- Derived quantity
- Two conventions:
  1. **Split**: E[i] = Σⱼ E(i,j)/2 → Sum = Total energy ✓
  2. **Double-count**: E[i] = Σⱼ E(i,j) → Sum = 2× Total energy

**When to use**:
- Visualization → Double-count (shows activity)
- Analysis → Split (correct thermodynamics)

### Energy Terms

**Bonded** (sequential neighbors):
- FENE: Backbone constraint
- BEXC: Prevent overlap
- STCK: Major stabilization (-1 to -2 per bond)

**Non-bonded**:
- NEXC: Excluded volume (repulsive)
- HB: Watson-Crick pairing (-0.5 to -1.5)
- CRSTCK, CXSTCK: Additional stacking
- DH: Electrostatic repulsion (salt-dependent)

### oxDNA2 vs oxDNA

**Key differences**:
- Stronger stacking (1.3523 vs 1.3448)
- Modified hydrogen bonding (1.0678 vs 1.077)
- Enhanced coaxial stacking (k=58.5 vs 46.0)
- **Salt-dependent Debye-Huckel** term (new in oxDNA2)
- Different FENE equilibrium (0.7564 vs 0.7525)

---

## File Cross-References

### Understanding Energy Calculation
1. Start: [ENERGY_ANALYSIS_GUIDE.md](ENERGY_ANALYSIS_GUIDE.md)
2. Details: [output_bonds_explanation.md](output_bonds_explanation.md)
3. Implementation: [oxdna2_pytorch.py](oxdna2_pytorch.py)
4. Examples: [test_oxdna2_pytorch.py](test_oxdna2_pytorch.py)

### Using PyTorch Implementation
1. Quick start: [README.md](README.md)
2. Technical details: [oxdna2_pytorch_readme.md](oxdna2_pytorch_readme.md)
3. Examples: [test_oxdna2_pytorch.py](test_oxdna2_pytorch.py)
4. Analysis guide: [ENERGY_ANALYSIS_GUIDE.md](ENERGY_ANALYSIS_GUIDE.md)

### Per-Nucleotide Analysis
1. Guide: [ENERGY_ANALYSIS_GUIDE.md](ENERGY_ANALYSIS_GUIDE.md)
2. Tool: [output_per_nucleotide_energy.py](output_per_nucleotide_energy.py)
3. Understanding: [output_bonds_explanation.md](output_bonds_explanation.md)

---

## Support

### Troubleshooting

**Issue**: Import errors
- Solution: Install dependencies: `pip install torch numpy oxDNA-analysis-tools`

**Issue**: Energies don't match original oxDNA
- Check: Are you using the same temperature and salt?
- Check: Are meshes enabled in C++ but not in PyTorch?
- Note: Small numerical differences expected (PyTorch computes f4 directly)

**Issue**: Total energy is 2× expected
- Likely: Using double-counting instead of split energies
- Solution: Use `output_per_nucleotide_energy.py` or divide by 2

### Getting Help

1. Read [ENERGY_ANALYSIS_GUIDE.md](ENERGY_ANALYSIS_GUIDE.md)
2. Check examples in [test_oxdna2_pytorch.py](test_oxdna2_pytorch.py)
3. Consult original oxDNA documentation: https://dna.physics.ox.ac.uk/

---

## Citation

If you use this code, please cite:

**Original oxDNA papers**:
1. T. E. Ouldridge, A. A. Louis, and J. P. K. Doye, "Structural, mechanical, and thermodynamic properties of a coarse-grained DNA model", *J. Chem. Phys.* **134**, 085101 (2011)

2. B. E. K. Snodin, F. Randisi, M. Mosayebi, et al., "Introducing improved structural properties and salt dependence into a coarse-grained model of DNA", *J. Chem. Phys.* **142**, 234901 (2015)

**This implementation**:
- PyTorch implementation by [Your Name], faithfully translated from the C++ oxDNA code

---

## License

Follows the same license as the original oxDNA code.

---

## Summary

This directory provides **three complementary approaches** to energy analysis:

1. **Original tool** (`output_bonds.py` in main repo)
   - Pros: Standard, well-tested, oxView compatible
   - Cons: Double-counts energies
   - Use for: Visualization

2. **Modified tool** (`output_per_nucleotide_energy.py`)
   - Pros: Correct accounting, oxView compatible
   - Cons: Requires manual installation
   - Use for: Quantitative analysis

3. **PyTorch implementation** (`oxdna2_pytorch.py`)
   - Pros: Flexible, fast, differentiable
   - Cons: Not drop-in replacement for C++ code
   - Use for: Custom analysis, machine learning

Choose the tool that best fits your needs!

---

**Last updated**: October 2024
**Total lines of code**: ~3000
**Total documentation**: ~2000 lines
**Test coverage**: 8 comprehensive tests
