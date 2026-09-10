# AGENTS.md — Task Tracker

> Auto-maintained by Claude Code. Edit manually or via `/agents-md` commands.
> To resume after interruption: run `/agents-md resume` in a new session.

## Current Status

**Last Updated:** 2026-09-09 18:40
**Last Session Summary:** _Precision tiers + edge list + CPU external forces done and pushed. Optimized DNA2 (threadgroup 128, sqr_rcut skin cull), fixed use_edge crash. `Metal_list = edge` gives ~20% over verlet on DNA2. Running the ErikPoppleton/oxDNA_performance benchmark matrix (128 – 524288 nuc × {CPU, float/mixed × verlet/edge, hardmixed}) for the plots._
**Resume From:** _Benchmark matrix running (scratchpad/bench_metal.sh → results.csv, plot_bench.py). Then commit benchmark script + plots. Open: #11 CPU OpenMP; #14 RNA/LJ native kernels; edge-kernel micro-opt (precompute per-particle axes)._

---

## Active Tasks

| ID | Task | Status | Started |
|----|------|--------|---------|
| #16 | Run ErikPoppleton/oxDNA_performance benchmarks (origami-ish duplex boxes 128–524288 nuc) across precision tiers + verlet/edge + CPU; ms/step, total time, plots | 🔄 In Progress | 2026-09-09 |
| #11 | OpenMP for the **CPU** MD/MC force loop — deferred (core refactor: stateful `_computed_r`/`_is_infinite`, half-list + 3rd-law races). Metal-side OpenMP done. | ⏸️ Deferred | 2026-09-09 |
| #14 | Native RNA / LJ / patchy / TEP GPU force kernels (CPU-fallback only today) | ⏳ Pending | 2026-09-09 |
| #15 | GPU barostat/NPT, stress tensor, external forces on GPU (currently CPU each step), particle sorting | ⏳ Pending | 2026-09-09 |
| #17 | Edge-kernel micro-opt: precompute per-particle axes once/step (get_axes is ~10-20% of per-edge cost, recomputed in both edge kernels) | ⏳ Pending | 2026-09-09 |

---

## Completed Tasks

| ID | Task | Completed |
|----|------|-----------|
| #1 | Pull `origin/master` into `metal` (was 148 commits behind); resolve conflicts; push | 2026-09-08 |
| #2 | Install Metal toolchain; build `-DMETAL=ON` | 2026-09-08 |
| #3 | Diagnose both run modes with `comparison_run/validate_metal_forcefields.py` | 2026-09-08 |
| #4 | Fix CPU-fallback path to rebuild Verlet lists only past the skin | 2026-09-08 |
| #5 | Port native DNA/DNA2 force+torque kernel faithfully from `CUDA_DNA.cuh` | 2026-09-09 |
| #6 | Batch the whole native MD step into one GPU submission (5 waits/step → 1) | 2026-09-09 |
| #7 | Phase A: `backend_precision` scaffolding (`float`/`mixed`/`hardmixed`); reject GPU `double` / `-DMETAL_DOUBLE` | 2026-09-09 |
| #8 | Phase B: `float` tier = the native path; dead `METAL_DOUBLE` shader typedef removed | 2026-09-09 |
| #9 | Phase C: `hardmixed` tier — GPU float forces + CPU double velocity-Verlet over shared buffers; `_sync_forces_torques_from_gpu`/`_sync_vels_Ls_from_gpu`; OpenMP on those loops | 2026-09-09 |
| #10 | Phase D: `mixed` tier — `src/Metal/Shaders/df64.h` double-float library + `first_step_mixed`/`second_step_mixed`/`mixed_sync_df_vels` kernels | 2026-09-09 |
| #12 | Energy validation for all three tiers (NVE + Brownian thermostat) vs CPU backend; regression run of `validate_metal_forcefields.py` (fallback 7/7, native DNA/DNA2 pass); best-of-3 benchmark table | 2026-09-09 |
| #13 | Documentation — `BUILD_METAL.md` rewrite (precision tiers, df64, benchmarks, OpenMP status), `src/Metal/README.md` refresh | 2026-09-09 |
| #18 | CPU external forces for native/mixed/hardmixed paths (traps/walls were silently dropped); TRAPS example now matches CPU. Commit `ac22deb7`. | 2026-09-09 |
| #19 | DNA2 kernel tuning: threadgroup 64→128, `sqr_rcut` skin-shell cull; `use_edge` accepted-and-ignored (was crashing examples/CUDA_EXAMPLE). Commit `ac22deb7`. | 2026-09-09 |
| #20 | `Metal_list = edge` — flat pair list + per-edge DNA force kernel (atomic scatter). ~20% over verlet on DNA2. Commit `9686df33`. | 2026-09-09 |

---

## Benchmarks (Apple M4, Metal_EXAMPLE = 32768 nucleotides, DNA interaction)

3000 NVE steps, best-of-3 wall-clock, vs the double-precision CPU backend (`-O3`):

| backend / tier               | E_tot @ 3000 | rel. err vs CPU | wall time | speed-up |
|------------------------------|--------------|-----------------|-----------|----------|
| CPU (`-DDOUBLE=ON`)          | -1.077815    | —               | 46.5 s    | 1.0x     |
| Metal `float`                | -1.077816    | 1e-6            | 3.05 s    | 15.2x    |
| Metal `mixed` (df64)         | -1.077817    | 2e-6            | 3.22 s    | 14.4x    |
| Metal `hardmixed`            | -1.077819    | 4e-6            | 9.63 s    | 4.8x     |
| Metal `hardmixed` + OpenMP 4t| -1.077819    | 4e-6            | 7.70 s    | 6.0x     |

Brownian thermostat (dt 0.002, newtonian_steps 53): all tiers hold <KE>/N ≈ 0.300 (= 6 · T/2).
Machine is noisy under load (CPU baseline ranged 46–1100 s); table uses fastest clean run.

---

## Session Log

### 2026-09-08
- Merged master (4 conflicts), Metal toolchain, all binaries built. CPU-fallback path correct (~1e-6) but slow; native kernels broken. Fixed fallback list frequency. Backups: branch `metal-backup-2026-09-08`, tag `metal-pre-merge-2026-09-08`.

### 2026-09-09
- Rewrote `dna_kernels.metal` as a faithful `CUDA_DNA.cuh` port (HB, cross/coax stacking, Debye-Hückel, FENE cap, body-frame torque). Native DNA/DNA2 match the CPU backend. `lists_are_old()` skin check. Batched the MD step into one GPU submission. Commits `77bc1ee1`, `580660ad`.
- Implemented the three precision tiers. `mixed` = double-float (df64: TwoSum, TwoProd-FMA, Dekker add — `src/Metal/Shaders/df64.h`) with df64 velocity-Verlet kernels; keeps float mirror buffers for the force kernels / lists / thermostats. `hardmixed` = GPU float forces + CPU native-double integration over the shared unified-memory buffers, OpenMP-parallel. `-DMETAL_DOUBLE` is now a configure error. `-DUSE_OPENMP` option added (libomp on macOS). Validated all tiers (NVE 1e-6/2e-6/4e-6; thermostat holds T). Commit `9d8ac5f9` pushed.
- **CPU-backend OpenMP investigation (#11):** the MD force loop is ~95% of CPU MD time, so only parallelizing it matters. Two blockers:
  1. `BaseInteraction::_computed_r` (and `_is_infinite`) are **mutable instance members** used as scratch between `pair_interaction_*` and its helpers — two threads sharing one interaction object race on them. Fix = make them `thread_local` / pass `r` explicitly.
  2. `BaseList::get_neigh_list(p)` is a **half list** (`q->index < p->index`) and `pair_interaction_nonbonded` applies Newton's 3rd law (writes `p->force` **and** `q->force`) — parallelizing over `p` races on `q->force`. Fix = full neighbour list + one-sided force output (API addition), per-thread force buffers + reduction, or cell colouring.
  Neither is a small change; oxDNA upstream has never OpenMP'd the core (it uses MPI for parallel tempering and CUDA for the GPU). MC is a sequential Markov chain — "multicore MC" means independent replicas / parallel tempering, which `PT_VMMC_CPUBackend` already provides via MPI.
  Recommendation: on Apple Silicon use the Metal `float`/`mixed` tiers (15x) rather than CPU threads. If CPU OpenMP is still wanted, it is a dedicated task: thread-local interaction state + per-thread force accumulation.
- Re-benchmarked best-of-3 after noticing the machine was under heavy load during the first pass (CPU baseline varied 46-1100 s). Clean numbers: CPU 46.5 s; float 3.05 s (15.2x); mixed 3.22 s (14.4x, +5% over float); hardmixed 9.63 s (4.8x) / 7.70 s with 4 OpenMP threads (6.0x, ~1.9x from threads, plateaus at 4). All tiers verified in the `-DUSE_OPENMP=ON` build too.
- Rewrote `BUILD_METAL.md` and `src/Metal/README.md`. Commits `9d8ac5f9` (precision tiers), `48c4e1b5` (docs) pushed.
- Session end: everything the user asked for is addressed except the CPU force-loop OpenMP (deferred, #11, documented). Metal precision tiers are the headline deliverable.
