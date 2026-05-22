# Agents.md — Edge List Implementation Tracking

**Author:** Subhajit Claude  
**Branch:** `edge`  
**Date:** 2026-05-22

---

## Summary

This branch adds a dedicated **edge list** neighbour-list implementation for both the CPU and GPU backends of oxDNA. An edge list stores each unique interacting pair (i, j) exactly once (with i > j), enabling Newton's-third-law force calculations without double-counting and with a cache-friendly flat iteration pattern.

---

## Files Added

### CPU

| File | Description |
|------|-------------|
| `src/Lists/EdgeList.h` | Header for the CPU `EdgeList` class |
| `src/Lists/EdgeList.cpp` | Implementation: flat `std::vector<ParticlePair>` with Verlet-skin lazy updates |

### GPU (CUDA)

| File | Description |
|------|-------------|
| `src/CUDA/Lists/CUDAEdgeList.h` | Header for the GPU `CUDAEdgeList` class |
| `src/CUDA/Lists/CUDAEdgeList.cu` | Implementation: two-pass (count → scan → fill) direct edge building |
| `src/CUDA/Lists/CUDA_edge.cuh` | CUDA kernels: `edge_fill_cells`, `count_edges`, `fill_edges` |

### Benchmarks

| File | Description |
|------|-------------|
| `benchmarks/` | Benchmark examples cloned/adapted from ErikPoppleton/oxDNA_performance |
| `benchmarks/run_benchmark.sh` | Script to run verlet vs edge for CPU and GPU at multiple system sizes |
| `benchmarks/results/` | Output directory for benchmark timing results |

---

## Files Modified

| File | Change |
|------|--------|
| `src/Lists/ListFactory.cpp` | Added `"edge"` → `EdgeList` entry; added `#include "EdgeList.h"` |
| `src/CUDA/Lists/CUDAListFactory.cu` | Added `"edge"` → `CUDAEdgeList` entry; added `#include "CUDAEdgeList.h"` |
| `src/CMakeLists.txt` | Added `Lists/EdgeList.cpp` and `CUDA/Lists/CUDAEdgeList.cu` to source lists |

---

## Design

### CPU `EdgeList`

The standard Verlet list stores per-particle neighbour lists `_lists[i]` = all j < i within the
cutoff. `get_neigh_list(p)` returns this per-particle vector, and each call to the force loop
iterates `for p: for q in neigh_list[p]` — effectively visiting each pair once.

`EdgeList` pre-computes a **flat** `std::vector<ParticlePair>` of all (p, q) pairs at update time.

Key advantages:
- `get_potential_interactions()` returns the pre-built flat vector in O(1) vs the base-class
  O(N × avg_neigh) scan.
- A single contiguous allocation → better cache locality when iterating over all pairs.
- Verlet-skin lazy update (same mechanism as `VerletList`): only rebuilds when any particle
  displaces by more than `verlet_skin`.

Usage in input file:
```
list_type = edge
verlet_skin = 0.5
```

### GPU `CUDAEdgeList`

The existing `CUDASimpleVerletList` with `use_edge = true` builds edges in two steps:
1. Build full `d_matrix_neighs` (N × max_neigh matrix) — **O(N × max_neigh)** memory.
2. `compress_matrix_neighs` extracts unique pairs from the matrix.

`CUDAEdgeList` **eliminates `d_matrix_neighs` entirely** using a two-pass approach:

```
Pass 1 (count_edges kernel):
  Each thread counts edges where IND > m within the 27-cell neighbourhood.
  Writes per-particle counts to d_edge_counts[IND].

Scan (thrust::exclusive_scan):
  Computes per-particle offsets d_edge_offsets[IND] = prefix sum of counts.
  d_edge_offsets[N] = total number of edges (N_edges).

Pass 2 (fill_edges kernel):
  Each thread fills d_edge_list[d_edge_offsets[IND]..] with its edges.
  Output is a compact array of edge_bond{from, to} pairs.
```

Memory savings:  
- Removes `d_matrix_neighs`: `N × max_neigh × sizeof(int)` bytes  
- Keeps only `d_edge_list`: `N_edges × sizeof(edge_bond)` ≈ `N × avg_neigh/2 × 8` bytes  
- For N=50,000, max_neigh=100: saves ~20 MB

Usage in input file:
```
CUDA_list = edge
use_edge = true
edge_n_forces = 2
verlet_skin = 0.5
```

---

## How to Use

### CPU

```ini
backend = CPU
sim_type = MD
list_type = edge
verlet_skin = 0.5
```

### GPU

```ini
backend = CUDA
sim_type = MD
CUDA_list = edge
use_edge = true
edge_n_forces = 2
verlet_skin = 0.5
```

The existing `CUDA_list = verlet` + `use_edge = true` combination still works unchanged.

---

## Benchmark Results

Benchmarks use DNA2 systems from [ErikPoppleton/oxDNA_performance](https://github.com/ErikPoppleton/oxDNA_performance).

System sizes: N=128, N=512, N=4096, N=32768 nucleotides  
Simulation: 10M MD steps, DNA2 interaction, T=20°C, salt=1.0M

| System | Backend | List | Steps/sec |
|--------|---------|------|-----------|
| *to be filled after runs* | | | |

See `benchmarks/results/` for raw output.

---

## Known Limitations

1. **CPU EdgeList + MC**: `get_neigh_list()` returns per-particle half-lists (compatible with MC).
   `get_potential_interactions()` returns the pre-built flat edge vector.
2. **GPU CUDAEdgeList + double precision**: Not supported (same constraint as `use_edge = true`).
3. **GPU CUDAEdgeList + barostat**: Not supported (same constraint as `use_edge = true`).
4. The edge list grows dynamically if the initial capacity estimate is exceeded (10% headroom added on resize).
