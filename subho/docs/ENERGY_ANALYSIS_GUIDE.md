# Guide to Energy Analysis in oxDNA

This guide explains how to compute and analyze energies in oxDNA simulations, with a focus on per-nucleotide energies.

## Quick Reference

| Tool | Output | Use Case |
|------|--------|----------|
| `output_bonds.py` | Pairwise energies | See all interactions, oxView visualization |
| `output_per_nucleotide_energy.py` | Per-nucleotide (split) | Quantitative analysis, correct totals |
| `oxdna2_pytorch.py` | Batch energy calculation | Machine learning, custom analysis |

---

## Understanding Energy Accounting

### Pairwise vs Per-Nucleotide Energies

In molecular simulations, energy is fundamentally **pairwise** - it describes interactions between pairs of particles.

**Example**: 3-nucleotide strand
```
0 ---[bond]--- 1 ---[bond]--- 2
```

**Pairwise energies**:
- E(0,1) = -2.0
- E(1,2) = -2.5
- **Total system energy = -4.5**

**Per-nucleotide energies** (two conventions):

1. **Split (correct accounting)**:
   ```
   E[0] = -1.0   (half of bond 0-1)
   E[1] = -2.25  (half of bond 0-1 + half of bond 1-2)
   E[2] = -1.25  (half of bond 1-2)

   Sum = -4.5 ✓ (matches total system energy)
   ```

2. **Double-count (visualization)**:
   ```
   E[0] = -2.0   (full bond 0-1)
   E[1] = -4.5   (full bond 0-1 + full bond 1-2)
   E[2] = -2.5   (full bond 1-2)

   Sum = -9.0 (twice the system energy!)
   Total = Sum / 2 = -4.5
   ```

### Which to Use?

- **oxView visualization**: Use double-count (original `output_bonds.py`)
  - Shows relative differences between nucleotides
  - Highlights "important" nucleotides with many interactions

- **Quantitative analysis**: Use split energies (`output_per_nucleotide_energy.py`)
  - Correct thermodynamic interpretation
  - Energy conservation
  - Comparable across different structures

---

## Tool 1: `output_bonds.py` (Original)

### What It Does

Computes **pairwise energies** between all interacting nucleotides.

### Usage

```bash
# Print pairwise energies to stdout
python -m oxDNA_analysis_tools.output_bonds input.dat trajectory.dat

# Generate per-nucleotide JSON files for oxView
python -m oxDNA_analysis_tools.output_bonds input.dat trajectory.dat -v output

# Use pN·nm units
python -m oxDNA_analysis_tools.output_bonds input.dat trajectory.dat -v output -u pNnm

# Parallel processing
python -m oxDNA_analysis_tools.output_bonds input.dat trajectory.dat -v output -p 4
```

### Output Format (Print Mode)

```
#id1 id2 FENE BEXC STCK NEXC HB CRSTCK CXSTCK DH total, t = 0
0 1 -2.3456 0.1234 -1.5678 0.0000 0.0000 0.0000 0.0000 0.0234 -3.7666
1 2 -2.2345 0.0987 -1.6543 0.0000 0.0000 0.0000 0.0000 0.0198 -3.7703
5 8 0.0000 0.0000 0.0000 0.1234 -0.8765 0.0000 0.0000 0.0876 -0.6655
```

**Columns**:
1. Particle 1 ID
2. Particle 2 ID
3. FENE (backbone)
4. BEXC (bonded excluded volume)
5. STCK (stacking)
6. NEXC (non-bonded excluded volume)
7. HB (hydrogen bonding)
8. CRSTCK (cross-stacking)
9. CXSTCK (coaxial stacking)
10. DH (Debye-Huckel, oxDNA2 only)
11. Total

### Output Format (Visualization Mode)

Creates JSON files for each energy term:
- `output_FENE.json`
- `output_STCK.json`
- `output_HB.json`
- etc.

**JSON structure**:
```json
{
"STCK (oxDNA su)" : [-1.234, -1.567, -1.890, ...]
}
```

Load in oxView: Overlay → Load file → Select JSON

### Energy Accounting

**Important**: In visualization mode, each pairwise energy is added to BOTH particles:

```python
energies[particle1] += E_pair
energies[particle2] += E_pair  # Full energy, not half!
```

This means:
- Sum of all per-nucleotide energies = **2× total system energy**
- Each nucleotide shows the sum of ALL its interactions
- Good for visualization (highlights active nucleotides)
- **Not** suitable for quantitative thermodynamic analysis

---

## Tool 2: `output_per_nucleotide_energy.py` (Modified, Correct Accounting)

### What It Does

Computes **per-nucleotide energies** with correct accounting (no double counting).

### Location

```bash
cd oxDNA/subho
python output_per_nucleotide_energy.py input.dat trajectory.dat -v output
```

### Usage

```bash
# Split energies (default, recommended for analysis)
python output_per_nucleotide_energy.py input.dat trajectory.dat -v output

# Use original double-counting behavior (same as output_bonds.py)
python output_per_nucleotide_energy.py input.dat trajectory.dat -v output --no-split

# With pN·nm units
python output_per_nucleotide_energy.py input.dat trajectory.dat -v output -u pNnm

# Parallel processing
python output_per_nucleotide_energy.py input.dat trajectory.dat -v output -p 4
```

### Key Difference

```python
# Original (output_bonds.py)
energies[p] += E_pair
energies[q] += E_pair

# Modified (output_per_nucleotide_energy.py)
energies[p] += E_pair / 2.0  # Split equally
energies[q] += E_pair / 2.0
```

### Output

Same format as `output_bonds.py`, but with correct energy accounting:
- Sum of per-nucleotide energies = **total system energy** (no factor of 2)
- Suitable for thermodynamic analysis
- Energy conservation guaranteed

**Additional output**:
```
Total system energy: -123.456 oxDNA su
Average per-nucleotide energy: -1.234 oxDNA su

Energy breakdown:
  FENE      :    -45.678 oxDNA su ( 37.0%)
  STCK      :    -67.890 oxDNA su ( 55.0%)
  HB        :     -9.888 oxDNA su (  8.0%)
  ...
```

---

## Tool 3: `oxdna2_pytorch.py` (Custom Analysis)

### What It Does

PyTorch implementation for batch energy calculations and custom analysis.

### Usage

```python
import torch
from oxdna2_pytorch import oxDNA2Energy

# Initialize model
model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

# Compute per-nucleotide energy
def compute_per_nucleotide_energies(positions, orientations, base_types,
                                    bonded_neighbors):
    """
    Compute per-nucleotide energies with custom accounting.

    Args:
        positions: (N, 3) tensor of positions
        orientations: (N, 3, 3) tensor of orientations
        base_types: (N,) tensor of base types
        bonded_neighbors: List[List[int]] - bonded neighbors for each nucleotide

    Returns:
        energies: (N, 8) tensor - per-nucleotide energies for each term
    """
    N = len(positions)
    energies = torch.zeros(N, 8)  # 8 energy terms

    # For each nucleotide
    for i in range(N):
        # Bonded interactions (only count 3' neighbor to avoid double counting)
        for j in bonded_neighbors[i]:
            if j > i:  # Only count each bond once
                # Compute bonded energies
                E_backbone = model.backbone_energy(...)
                E_stacking = model.stacking_energy(...)
                E_bexc = model.bonded_excluded_volume(...)

                # Split equally
                energies[i, 0] += E_backbone / 2.0
                energies[j, 0] += E_backbone / 2.0
                energies[i, 2] += E_stacking / 2.0
                energies[j, 2] += E_stacking / 2.0
                # etc.

        # Non-bonded interactions
        for j in range(N):
            if not is_bonded(i, j) and j > i:  # Avoid double counting
                E_hb = model.hydrogen_bonding_energy(...)
                E_nexc = model.nonbonded_excluded_volume(...)
                E_dh = model.debye_huckel_energy(...)
                # etc.

                # Split equally
                energies[i, 4] += E_hb / 2.0
                energies[j, 4] += E_hb / 2.0
                # etc.

    return energies

# Example: Compute energies for a trajectory
all_energies = []
for frame in trajectory:
    E = compute_per_nucleotide_energies(frame.positions, frame.orientations,
                                       frame.base_types, frame.bonded_neighbors)
    all_energies.append(E)

# Average over trajectory
mean_energies = torch.stack(all_energies).mean(dim=0)

# Total system energy
total_energy = mean_energies.sum()
```

### Advantages

1. **Flexible**: Custom energy accounting schemes
2. **Fast**: Batch processing, GPU support
3. **Automatic differentiation**: Get forces for free
4. **Machine learning**: Direct integration with PyTorch models

---

## Understanding Energy Terms

### Bonded Interactions (Sequential Neighbors)

| Term | Description | Typical Value | Favorable? |
|------|-------------|---------------|------------|
| FENE | Backbone connectivity | -2 to -3 | Yes (constraint) |
| BEXC | Bonded excluded volume | 0 to 0.2 | Repulsive |
| STCK | Base stacking | -1 to -2 | Yes (major stabilization) |

**Total bonded energy**: -3 to -5 per bond

### Non-Bonded Interactions

| Term | Description | Typical Value | Favorable? |
|------|-------------|---------------|------------|
| NEXC | Non-bonded excluded volume | 0 to 0.5 | Repulsive |
| HB | Hydrogen bonding (Watson-Crick) | -0.5 to -1.5 | Yes (base pairing) |
| CRSTCK | Cross-stacking | -0.1 to -0.5 | Sometimes |
| CXSTCK | Coaxial stacking | -0.1 to -0.3 | Sometimes |
| DH | Debye-Huckel (electrostatic) | 0 to 0.3 | Repulsive |

**Total non-bonded energy**: Highly variable depending on structure

---

## Common Analysis Tasks

### Task 1: Find Unstable Nucleotides

```python
# Compute per-nucleotide energies
energies, potentials = output_per_nucleotide_energy(
    traj_info, top_info, inputfile, visualize=True, split_energies=True
)
energies /= traj_info.nconfs

# Find nucleotides with high (unfavorable) total energy
total_col = -1  # Last column
threshold = np.mean(energies[:, total_col]) + 2*np.std(energies[:, total_col])
unstable = np.where(energies[:, total_col] > threshold)[0]

print(f"Unstable nucleotides: {unstable}")
```

### Task 2: Compare Two Structures

```python
# Structure 1
E1, _ = output_per_nucleotide_energy(traj_info1, top_info, inputfile,
                                     visualize=True, split_energies=True)
E1 /= traj_info1.nconfs

# Structure 2
E2, _ = output_per_nucleotide_energy(traj_info2, top_info, inputfile,
                                     visualize=True, split_energies=True)
E2 /= traj_info2.nconfs

# Energy difference
dE = E2 - E1

# Which nucleotides changed most?
biggest_changes = np.argsort(np.abs(dE[:, -1]))[-10:]
print(f"Top 10 changed nucleotides: {biggest_changes}")
```

### Task 3: Stacking vs Hydrogen Bonding Contribution

```python
energies, potentials = output_per_nucleotide_energy(...)

# Find column indices
stck_idx = potentials.index('STCK')
hb_idx = potentials.index('HB')

total_stacking = np.sum(energies[:, stck_idx])
total_hb = np.sum(energies[:, hb_idx])

print(f"Stacking contribution: {total_stacking:.2f} ({total_stacking/total_hb:.1f}× HB)")
print(f"H-bonding contribution: {total_hb:.2f}")
```

### Task 4: Energy Per Strand

```python
# Assuming you know which nucleotides belong to which strand
strand1_indices = np.arange(0, 24)
strand2_indices = np.arange(24, 48)

E_strand1 = np.sum(energies[strand1_indices, -1])
E_strand2 = np.sum(energies[strand2_indices, -1])

print(f"Strand 1 energy: {E_strand1:.2f}")
print(f"Strand 2 energy: {E_strand2:.2f}")
```

---

## Visualization in oxView

### Step 1: Generate JSON Files

```bash
python output_per_nucleotide_energy.py input.dat trajectory.dat -v mystructure
```

This creates:
- `mystructure_FENE.json`
- `mystructure_STCK.json`
- `mystructure_HB.json`
- etc.

### Step 2: Load in oxView

1. Open https://sulcgroup.github.io/oxdna-viewer/
2. Load your configuration and topology
3. Click **Overlay** → **Load file**
4. Select a JSON file (e.g., `mystructure_STCK.json`)
5. Nucleotides are colored by energy value

**Color interpretation**:
- Blue/Green: Low (favorable) energy
- Yellow/Orange: Medium energy
- Red: High (unfavorable) energy

### Step 3: Compare Multiple Energy Terms

You can load multiple JSON files simultaneously to compare different energy contributions.

---

## Converting Between Units

### oxDNA Simulation Units → pN·nm

```python
E_pNnm = E_oxdna * 41.42
```

### oxDNA Simulation Units → kcal/mol

```python
E_kcalmol = E_oxdna * 0.59  # Approximate at T=300K
```

### Temperature Conversion

```python
T_kelvin = T_oxdna * 3000
T_celsius = T_kelvin - 273.15
```

**Example**:
- T = 0.1 (sim units) = 300 K = 26.85°C (room temperature)
- T = 0.12 (sim units) = 360 K = 86.85°C (denaturing)

---

## Troubleshooting

### Problem: Energies seem too high

**Check**:
1. Are you using the correct units? (oxDNA su vs pN·nm)
2. Is the structure relaxed? (Unrelaxed structures have high excluded volume)
3. Are bonds broken? (FENE > 10 indicates broken backbone)

### Problem: Total energy doesn't match

**Check**:
1. Are you using split energies or double-counting?
2. Split: Sum = Total
3. Double-count: Sum = 2× Total

### Problem: oxView colors look wrong

**Check**:
1. Ensure JSON format is correct
2. Check that JSON contains correct number of values (one per nucleotide)
3. Try loading in a text editor to verify format

---

## References

- [output_bonds.py](../analysis/src/oxDNA_analysis_tools/output_bonds.py) - Original implementation
- [PairEnergy.cpp](../src/Observables/PairEnergy.cpp) - C++ observable
- [oxdna2_pytorch.py](oxdna2_pytorch.py) - PyTorch implementation
- [output_bonds_explanation.md](output_bonds_explanation.md) - Detailed explanation
- oxView: https://sulcgroup.github.io/oxdna-viewer/
- oxDNA documentation: https://dna.physics.ox.ac.uk/

---

## Summary

| Need | Use | Split Energies? |
|------|-----|-----------------|
| Pretty pictures in oxView | `output_bonds.py` | No (double-count) |
| Thermodynamic analysis | `output_per_nucleotide_energy.py` | Yes |
| Machine learning | `oxdna2_pytorch.py` | Your choice |
| Understand code | `output_bonds_explanation.md` | N/A |

**Remember**:
- Visualization → Use double-counting (shows relative importance)
- Analysis → Use split energies (correct thermodynamics)
