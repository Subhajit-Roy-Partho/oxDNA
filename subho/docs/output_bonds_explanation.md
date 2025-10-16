# Understanding `output_bonds.py` and Per-Nucleotide Energies

## Overview

The `output_bonds.py` script computes **pairwise interaction energies** between nucleotides in an oxDNA trajectory. It can either:
1. Print all pairwise energies to stdout
2. Generate per-nucleotide average energies for visualization in oxView

## How It Works

### 1. **Uses oxpy (Python bindings for oxDNA)**

The script uses the `oxpy` library to:
- Load the simulation configuration
- Create an analysis backend
- Use the `pair_energy` observable to compute energies

```python
inp["analysis_data_output_1"] = '''{
    name = stdout
    print_every = 1e10
    col_1 = {
        id = my_obs
        type = pair_energy  # <-- This is the key observable
    }
}'''
```

### 2. **The `pair_energy` Observable**

Located in [src/Observables/PairEnergy.cpp](../src/Observables/PairEnergy.cpp), this observable:

- Iterates through all **neighbor pairs** (particles within interaction range)
- Computes each energy term for each pair (FENE, stacking, H-bonding, etc.)
- Outputs: `particle_i particle_j E_fene E_bexc E_stack E_nexc E_hb E_crstack E_cxstack [E_DH] E_total`

**Example output format**:
```
#id1 id2 FENE BEXC STCK NEXC HB CRSTCK CXSTCK DH total, t = 0
0 1 -2.3456 0.1234 -1.5678 0.0000 0.0000 0.0000 0.0000 0.0234 -3.7666
1 2 -2.2345 0.0987 -1.6543 0.0000 0.0000 0.0000 0.0000 0.0198 -3.7703
5 8 0.0000 0.0000 0.0000 0.1234 -0.8765 0.0000 0.0000 0.0876 -0.6655
```

### 3. **Energy Terms in oxDNA2**

The script recognizes 8 (or 9) energy terms:

| Index | Name | Description | Interaction Type |
|-------|------|-------------|------------------|
| 0 | FENE | Backbone connectivity | Bonded |
| 1 | BEXC | Bonded excluded volume | Bonded |
| 2 | STCK | Stacking | Bonded |
| 3 | NEXC | Non-bonded excluded volume | Non-bonded |
| 4 | HB | Hydrogen bonding | Non-bonded |
| 5 | CRSTCK | Cross-stacking | Non-bonded |
| 6 | CXSTCK | Coaxial stacking | Non-bonded |
| 7 | DH | Debye-Huckel (oxDNA2 only) | Non-bonded |
| 8 | Total | Sum of all terms | - |

**Key comment from the code (lines 58-67)**:
```python
# The 9 energies in oxDNA2 are:
# 0 fene
# 1 bexc
# 2 stack
# 3 nexc
# 4 hb
# 5 cr_stack
# 6 cx_stack
# 7 Debye-Huckel <- this one is missing in oxDNA1
# 8 total
```

### 4. **Computing Per-Nucleotide Energies**

When `visualize=True`, the script converts **pairwise energies** to **per-nucleotide energies**:

```python
for e in e_txt[1:]:  # For each pairwise interaction
    e = e.split()
    p = int(e[0])  # Particle 1
    q = int(e[1])  # Particle 2
    l = np.array([float(x) for x in e[2:]])*conversion_factor  # Energy terms

    # Add HALF the energy to each particle
    energies[p] += l  # <-- Add to particle p
    energies[q] += l  # <-- Add to particle q
```

**Important**: Each pairwise energy is added to **both** particles!

### 5. **Why Add to Both Particles?**

This is the standard convention for distributing pairwise energies:
- A pair interaction E(i,j) involves both particles i and j
- To get "per-particle" energy, we split it equally: E(i) += E(i,j) and E(j) += E(i,j)
- This ensures that summing all per-particle energies gives **twice** the total system energy

**Mathematical relationship**:
```
Total system energy = (1/2) * Σ_i E_per_nucleotide(i)
                    = Σ_pairs E_pair(i,j)
```

The factor of 1/2 appears because each pair is counted twice when summing over particles.

### 6. **Averaging Over Trajectory**

When processing multiple frames:
```python
energies /= traj_info.nconfs  # Average over all configurations
```

This gives the **time-averaged per-nucleotide energy**.

## How to Use `output_bonds.py`

### Mode 1: Print Pairwise Energies

```bash
python output_bonds.py input.dat trajectory.dat
```

**Output**: Prints to stdout
```
#id1 id2 FENE BEXC STCK NEXC HB CRSTCK CXSTCK DH total, t = 0
0 1 -2.3456 0.1234 -1.5678 0.0000 0.0000 0.0000 0.0000 0.0234 -3.7666
1 2 -2.2345 0.0987 -1.6543 0.0000 0.0000 0.0000 0.0000 0.0198 -3.7703
...
```

### Mode 2: Generate Per-Nucleotide Energies for oxView

```bash
python output_bonds.py input.dat trajectory.dat -v output
```

**Output**: Creates separate JSON files for each energy term
- `output_FENE.json`
- `output_BEXC.json`
- `output_STCK.json`
- `output_NEXC.json`
- `output_HB.json`
- `output_CRSTCK.json`
- `output_CXSTCK.json`
- `output_DH.json` (oxDNA2 only)
- `output_total.json`

Each JSON file contains:
```json
{
"STCK (oxDNA su)" : [-1.234, -1.567, -1.890, ...]
}
```

These can be loaded in oxView to color nucleotides by energy.

### Additional Options

```bash
# Use pN·nm units instead of oxDNA simulation units
python output_bonds.py input.dat trajectory.dat -v output -u pNnm

# Use multiple CPUs for faster processing
python output_bonds.py input.dat trajectory.dat -v output -p 4

# Quiet mode (suppress info messages)
python output_bonds.py input.dat trajectory.dat -v output -q
```

## Getting True Per-Nucleotide Energies

### Current Behavior

The current implementation adds the **full pairwise energy** to both particles:

```python
energies[p] += E_pair(p,q)  # Full energy to p
energies[q] += E_pair(p,q)  # Full energy to q
```

This means each nucleotide's energy includes all its interactions with neighbors, but each interaction is **double-counted** in the total.

### If You Want Individual Contributions

To get the actual energy contribution of each nucleotide (without double counting), you would need to:

**Option 1: Divide by 2 (Post-processing)**
```python
energies_per_nucleotide = energies / 2.0
```

But this loses information about which interactions contribute to each nucleotide.

**Option 2: Modify the Script**

You could modify the script to assign energy differently:

```python
# Option A: Split energy equally between particles
energies[p] += l / 2.0  # Half to particle p
energies[q] += l / 2.0  # Half to particle q

# Option B: Assign based on interaction type
if is_bonded(p, q):
    energies[p] += l / 2.0  # Split bonded interactions
    energies[q] += l / 2.0
else:
    # Non-bonded: could assign based on geometric criteria
    # or keep the full double-counting
    energies[p] += l
    energies[q] += l
```

**Option 3: Use a Custom Observable**

Create a new observable in C++ that computes true per-particle energies:

```cpp
// In a new ParticleEnergy.cpp observable
for (int i = 0; i < N; i++) {
    BaseParticle *p = particles[i];
    number E_particle = 0;

    // Bonded interactions (only count if p is the 3' neighbor)
    if (p->n5 != P_VIRTUAL) {
        E_particle += interaction->pair_interaction_bonded(p->n5, p);
    }

    // Non-bonded interactions (count all neighbors)
    std::vector<BaseParticle*> neighbors = lists->get_neigh_list(p);
    for (BaseParticle *q : neighbors) {
        if (!p->is_bonded(q)) {
            E_particle += interaction->pair_interaction_nonbonded(p, q) / 2.0;
        }
    }

    output << E_particle << " ";
}
```

## Example: Modified Script for Half Energies

Here's how you could modify `output_bonds.py` to split energies:

```python
# In the compute() function, around lines 74-81:
if ctx.visualize:
    for e in e_txt[1:]:
        if not e[0] == '#':
            e = e.split()
            p = int(e[0])
            q = int(e[1])
            l = np.array([float(x) for x in e[2:]])*ctx.conversion_factor

            # MODIFICATION: Split energy equally
            energies[p] += l / 2.0  # Half to particle p
            energies[q] += l / 2.0  # Half to particle q
```

**Effect**: Now summing all per-nucleotide energies gives the total system energy (not double).

## Understanding the Energy Distribution

### Example: A 3-Nucleotide System

```
Nucleotides: 0---1---2  (bonded)
```

**Pairwise energies**:
- E(0,1) = -2.0 (bonded: FENE + stacking)
- E(1,2) = -2.5 (bonded: FENE + stacking)

**Current implementation** (full double-counting):
```
E[0] = -2.0  (from pair 0-1)
E[1] = -2.0 - 2.5 = -4.5  (from pairs 0-1 and 1-2)
E[2] = -2.5  (from pair 1-2)

Total = -2.0 - 4.5 - 2.5 = -9.0
Actual total = -2.0 - 2.5 = -4.5  (factor of 2!)
```

**Modified implementation** (split energies):
```
E[0] = -1.0  (half of pair 0-1)
E[1] = -1.0 - 1.25 = -2.25  (half of each pair)
E[2] = -1.25  (half of pair 1-2)

Total = -1.0 - 2.25 - 1.25 = -4.5  ✓ (correct!)
```

## Using PyTorch Implementation for Per-Nucleotide Energy

You can also use our PyTorch implementation to compute per-nucleotide energies:

```python
import torch
from oxdna2_pytorch import oxDNA2Energy

model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

# For each nucleotide i
per_nucleotide_energy = []

for i in range(N):
    E_i = 0.0

    # Bonded interactions
    if has_5prime_neighbor[i]:
        E_i += model.stacking_energy(pos[i-1:i], pos[i:i+1],
                                     ori[i-1:i], ori[i:i+1])
        E_i += model.backbone_energy(r_backbone[i])
        E_i += model.bonded_excluded_volume(...)

    # Non-bonded interactions (split equally)
    for j in neighbors[i]:
        if not bonded(i, j):
            E_ij = model.hydrogen_bonding_energy(...) + \
                   model.cross_stacking_energy(...) + \
                   model.coaxial_stacking_energy(...) + \
                   model.nonbonded_excluded_volume(...) + \
                   model.debye_huckel_energy(...)
            E_i += E_ij / 2.0  # Split non-bonded

    per_nucleotide_energy.append(E_i)
```

## Summary

| Method | Double Counts? | Use Case |
|--------|----------------|----------|
| Current `output_bonds.py` | Yes (factor of 2) | Visualization in oxView (shows relative values) |
| Modified with `/2.0` | No | Analysis, thermodynamic calculations |
| Custom C++ observable | No | Maximum efficiency, full control |
| PyTorch implementation | User controlled | Machine learning, analysis |

**Key takeaway**: The current implementation is designed for **visualization** where relative energies matter more than absolute values. For quantitative analysis, you should divide by 2 or use a modified version.

## References

- [PairEnergy.cpp](../src/Observables/PairEnergy.cpp) - C++ implementation
- [output_bonds.py](./output_bonds.py) - Python wrapper
- oxView documentation: https://sulcgroup.github.io/oxdna-viewer/
