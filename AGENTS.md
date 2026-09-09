# AGENTS.md — Task Tracker

> Auto-maintained by Claude Code. Edit manually or via `/agents-md` commands.
> To resume after interruption: run `/agents-md resume` in a new session.

## Current Status

**Last Updated:** 2026-09-09 16:40
**Last Session Summary:** _Metal `float`/`mixed`/`hardmixed` precision tiers implemented, validated against the CPU backend (E match 1e-6 / 2e-6 / 4e-6) and committed/pushed. Optional `-DUSE_OPENMP` wired for the Metal CPU-integration loops. Investigating OpenMP for the CPU MD/MC force loop._
**Resume From:** _#11 — CPU-backend OpenMP is blocked by two architectural issues (see Session Log 2026-09-09); decide whether to do the core refactor. Then #13 documentation._

---

## Active Tasks

| ID | Task | Status | Started |
|----|------|--------|---------|
| #11 | OpenMP for the **CPU** MD/MC force loop — blocked by core architecture (stateful `BaseInteraction::_computed_r`; half neighbour lists + Newton's-3rd-law force races). Metal-side OpenMP is done. Needs a decision on the invasive refactor. | ⏸️ Paused | 2026-09-09 |
| #12 | Extend the benchmark: smaller system + longer runs, `mixed` df64 drift over 1e5+ steps, a table in the docs | 🔄 In Progress | 2026-09-09 |
| #13 | Documentation — `BUILD_METAL.md` precision-mode section, `src/Metal/README.md`, df64 notes, benchmark table, OpenMP status | ⏳ Pending | 2026-09-09 |

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
| #12a | Energy validation for all three tiers (NVE + Brownian thermostat) vs CPU backend; regression run of `validate_metal_forcefields.py` (fallback 7/7, native DNA/DNA2 pass) | 2026-09-09 |

---

## Benchmarks (Apple M4, Metal_EXAMPLE = 32768 nucleotides, DNA interaction)

3000 NVE steps, wall-clock, vs the double-precision CPU backend:

| backend / tier          | E_tot @ 3000 | rel. err vs CPU | wall time | speed-up |
|-------------------------|--------------|-----------------|-----------|----------|
| CPU (`-DDOUBLE=ON`)     | -1.077815    | —               | 131.5 s   | 1.0x     |
| Metal `float`           | -1.077816    | 1e-6            | 10.4 s    | 12.6x    |
| Metal `mixed` (df64)    | -1.077817    | 2e-6            | 12.1 s    | 10.9x    |
| Metal `hardmixed`       | -1.077819    | 4e-6            | 25.0 s    | 5.3x     |

Brownian thermostat (dt 0.002, newtonian_steps 53): all tiers hold <KE>/N ≈ 0.300 (= 6 · T/2).

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
  Recommendation: on Apple Silicon use the Metal `float`/`mixed` tiers (10-12x) rather than CPU threads. If CPU OpenMP is still wanted, it is a dedicated task: thread-local interaction state + per-thread force accumulation.
- Stopped at: writing documentation (#13).
