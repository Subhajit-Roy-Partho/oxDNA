# Metal backend for oxDNA (Apple Silicon)

The Metal backend runs oxDNA molecular dynamics on Apple GPUs (M-series). It is
the Apple-GPU analogue of the CUDA backend: the force/torque evaluation, the
Verlet-list build and the velocity-Verlet integration all run on the GPU.

- **Working today:** MD with `interaction_type = DNA` and `DNA2` (oxDNA1 and
  oxDNA2), including hydrogen bonding, stacking, cross-/coaxial stacking,
  excluded volume, FENE with the `max_backbone_force` cap, and Debye-Hückel.
  The Brownian (`brownian` / `john`) thermostat.
- **CPU fallback:** every other interaction (RNA, LJ, patchy, TEP, …) runs
  through a CPU force fallback — correct but not faster than the CPU backend.
- **Not ported:** MC/VMMC, barostat/NPT, FFS, external forces on the GPU,
  the stress tensor.

---

## Build

### Prerequisites

- macOS with an Apple-Silicon GPU
- Xcode + Command Line Tools
- The Metal shader toolchain. If `xcrun -sdk macosx metal --version` fails:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
- CMake ≥ 3.5, Ninja
- (optional) `libomp` for `-DUSE_OPENMP=ON`: `brew install libomp`

### Configure and build

```bash
cd oxDNA
cmake -S . -B build_metal -G Ninja -DMETAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_metal -j8
```

Executables land in `build_metal/bin/` (`oxDNA`, `DNAnalysis`, `confGenerator`)
together with `shaders.metallib`, which the executable loads at run time from
its own directory or from `src/Metal/build/`.

### Build options

| option | meaning |
|--------|---------|
| `-DMETAL=ON` | enable the Metal backend (mutually exclusive with `-DCUDA=ON`) |
| `-DUSE_OPENMP=ON` | multithread the CPU-side integration loops used by the `hardmixed` tier and the CPU fallback (needs an OpenMP runtime) |
| ~~`-DMETAL_DOUBLE=ON`~~ | **removed** — Apple GPUs have no hardware double precision. Configuring with it is a hard error. Choose accuracy at run time with `backend_precision` (below). |

---

## Running

Minimal input file:

```
backend = Metal
Metal_avoid_cpu_calculations = 1     # 1 = native GPU kernels, 0 = CPU force fallback
backend_precision = float            # float | mixed | hardmixed  (see below)

sim_type = MD
interaction_type = DNA               # or DNA2
T = 300K
dt = 0.003
thermostat = brownian
diff_coeff = 2.5
newtonian_steps = 53
verlet_skin = 0.1
max_backbone_force = 10.0            # recommended: caps the FENE force

steps = 1000000
conf_file = init.conf
topology = init.top
trajectory_file = trajectory.dat
energy_file = energy.dat
```

### `Metal_avoid_cpu_calculations`

- `1` — native GPU force kernels. Use this for `DNA`/`DNA2`.
- `0` — CPU force fallback: positions are copied to the CPU each step, the
  normal CPU interaction computes the forces, and the result is copied back.
  Physically identical to the CPU backend (≈1e-6 relative energy error) but no
  faster. This is the only correct option for RNA/LJ/patchy/TEP today.

### `backend_precision`

Apple GPUs cannot do `double` in a shader at all, so the CUDA `double` and
`mixed` kernels have no direct port. The three tiers below are the achievable
analogues; all of them evaluate **forces in float32**.

| tier | positions / velocities / L | integration | when to use |
|------|----------------------------|-------------|-------------|
| `float` (default) | float32 | float32 velocity-Verlet on the GPU | default; fastest |
| `mixed` | double-float (df64: a pair of float32, ~48-bit mantissa) | df64 velocity-Verlet on the GPU | long runs where float position round-off matters; ~15 % slower than `float` |
| `hardmixed` | native `double` on the CPU | `double` velocity-Verlet on the CPU over the shared unified-memory buffers (OpenMP-parallel with `-DUSE_OPENMP=ON`) | maximum integration accuracy; ~2.4× slower than `float` |

`backend_precision = double` is rejected with a message pointing at `mixed` /
`hardmixed`.

**How `mixed` works.** Positions, velocities and angular momenta are stored as
`hi`/`lo` float pairs (`src/Metal/Shaders/df64.h`: error-free `TwoSum`, FMA-based
`TwoProd`, Dekker addition — see Thall, *Extended-precision floating-point
numbers for GPU computation*, 2006). The `first_step_mixed` / `second_step_mixed`
kernels integrate in df64 and write a plain-float "mirror" of each quantity that
the force kernels, the Verlet-list check and the thermostat read. When the GPU
thermostat rescales velocities, `mixed_sync_df_vels` re-splits the mirror back
into df64.

**How `hardmixed` works.** Each step: the GPU computes float forces/torques into
shared buffers; the CPU reads them straight out of unified memory (no copy),
runs the velocity-Verlet half-kicks and the quaternion update in `double`, and
writes float positions/orientations back for the next force evaluation. The GPU
Verlet lists are rebuilt on the host displacement check, exactly like `float`.

---

## Accuracy and performance

Apple M4, `Metal_EXAMPLE` (32 768 nucleotides, `interaction_type = DNA`),
3000 NVE steps, wall-clock, against the `-DDOUBLE=ON` CPU backend:

| backend / tier | E_tot after 3000 steps | rel. error vs CPU | wall time | speed-up |
|----------------|------------------------|-------------------|-----------|----------|
| CPU (double)   | −1.077815              | —                 | 131.5 s   | 1.0×     |
| Metal `float`  | −1.077816              | 1 × 10⁻⁶          | 10.4 s    | **12.6×** |
| Metal `mixed`  | −1.077817              | 2 × 10⁻⁶          | 12.1 s    | **10.9×** |
| Metal `hardmixed` | −1.077819           | 4 × 10⁻⁶          | 25.0 s    | **5.3×**  |

With the Brownian thermostat (`dt = 0.002`, `newtonian_steps = 53`) every tier
holds ⟨KE⟩/N ≈ 0.300 (= 6 · T/2 for a rigid nucleotide). Trajectories diverge
from the CPU run only through the RNG stream (the GPU thermostat uses a
per-particle PCG generator, the CPU one a single `drand48` stream) — this is
expected for a stochastic simulation.

Validate on your own machine:

```bash
cmake -S . -B build_cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DDOUBLE=ON
cmake --build build_cpu -j8 --target oxDNA confGenerator
cd comparison_run
python3 validate_metal_forcefields.py --metal-avoid-cpu-calculations 1 --scenarios dna dna2
python3 validate_metal_forcefields.py --metal-avoid-cpu-calculations 0   # all forcefields, fallback
```

---

## Implementation notes

| file | contents |
|------|----------|
| `src/Metal/Backends/MD_MetalBackend.mm` | the MD driver: per-step command-buffer batching, the three precision paths, the CPU-fallback / hardmixed path |
| `src/Metal/Shaders/dna_kernels.metal` | DNA/DNA2 force + torque kernel — a line-by-line port of `src/CUDA/Interactions/CUDA_DNA.cuh`. Torque is returned in the particle body frame. |
| `src/Metal/Shaders/md_kernels.metal` | float and df64 velocity-Verlet kernels |
| `src/Metal/Shaders/df64.h` | double-float arithmetic |
| `src/Metal/Lists/MetalSimpleVerletList.mm` | GPU cell + Verlet-list build; `lists_are_old()` host-side skin check |
| `src/Metal/Interactions/MetalCPUForceFallback.mm` | CPU force fallback |

### Why the whole step is one command buffer

The native `float` and `mixed` paths encode `first_step → zero → forces →
second_step → thermostat` into a **single** `MTLCommandBuffer` with one
`waitUntilCompleted`. An earlier version issued five separate command buffers
per step, each with its own blocking wait, and was dominated by command-buffer
round-trip latency rather than GPU compute (≈5× slower).

### OpenMP / CPU multithreading

`-DUSE_OPENMP=ON` multithreads the CPU-side per-particle loops in the Metal
backend (`hardmixed` integration, buffer packing, the force/velocity sync-backs).

The **CPU MD/MC backends themselves are still single-threaded.** Their hot loop
is the pairwise force sum (~95 % of CPU MD time), and parallelizing it correctly
needs core changes that are out of scope here:

1. `BaseInteraction::_computed_r` / `_is_infinite` are mutable scratch members
   shared between `pair_interaction_*` and its helpers — threads sharing one
   interaction object race on them.
2. `BaseList::get_neigh_list(p)` is a half list and `pair_interaction_nonbonded`
   applies Newton's third law (writes `p->force` **and** `q->force`), so a loop
   over `p` races on `q->force`.

A correct implementation needs thread-local interaction scratch plus either a
full neighbour list with one-sided force output, per-thread force buffers with a
reduction, or cell colouring. On Apple Silicon, prefer the Metal `float` /
`mixed` tiers (10–12×) over CPU threads. Monte Carlo is a sequential Markov
chain; "multicore MC" means independent replicas / parallel tempering, which
`PT_VMMC_CPUBackend` already provides through MPI.
