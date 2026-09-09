# AGENTS.md — Task Tracker

> Auto-maintained by Claude Code. Edit manually or via `/agents-md` commands.
> To resume after interruption: run `/agents-md resume` in a new session.

## Current Status

**Last Updated:** 2026-09-09 12:29
**Last Session Summary:** _Merged master into `metal`, ported the native DNA/DNA2 GPU force kernel from CUDA, batched the MD step — native path is now ~9× faster than the CPU backend and matches its energies. Starting the precision-tier + multicore work._
**Resume From:** _Phase A — introduce `backend_precision` handling (`float` / `hardmixed` / `mixed`) in `src/Metal/Backends/MD_MetalBackend.mm` and `src/Metal/Shaders/dna_kernels.metal`._

---

## Active Tasks

| ID | Task | Status | Started |
|----|------|--------|---------|
| #7 | Phase A: precision-mode scaffolding — `backend_precision` = `float` / `hardmixed` / `mixed`; reject GPU `double` with guidance (Metal has no hardware double) | 🔄 In Progress | 2026-09-09 |
| #8 | Phase B: `float` mode — formalize the current native path as the float precision class; clean up the dead `METAL_DOUBLE` shader typedef | ⏳ Pending | 2026-09-09 |
| #9 | Phase C: `hardmixed` mode — GPU float forces + CPU double velocity-Verlet over shared (unified-memory) buffers; OpenMP-parallel CPU integration; tune the energy accumulation | ⏳ Pending | 2026-09-09 |
| #10 | Phase D: `mixed` mode — double-float (df64, Thall 2006 / Dekker / Knuth two-sum) arithmetic in MSL; df64 positions + integration; optimize | ⏳ Pending | 2026-09-09 |
| #11 | Phase E: investigate + prototype OpenMP for the CPU MD (and MC) backends — currently single-threaded; document feasibility, implement force loop if safe | ⏳ Pending | 2026-09-09 |
| #12 | Phase F: full CPU-vs-GPU energy comparison for `float` / `hardmixed` / `mixed`; benchmark + report speed-ups for every case | ⏳ Pending | 2026-09-09 |
| #13 | Phase G: detailed documentation — update `BUILD_METAL.md` / `src/Metal/README.md`, precision-mode guide, benchmark table | ⏳ Pending | 2026-09-09 |

---

## Completed Tasks

| ID | Task | Completed |
|----|------|-----------|
| #1 | Pull `origin/master` into `metal` (was 148 commits behind); resolve conflicts; push | 2026-09-08 |
| #2 | Install Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`); build `-DMETAL=ON` | 2026-09-08 |
| #3 | Diagnose both run modes with `comparison_run/validate_metal_forcefields.py` (fallback correct-but-slow; native broken) | 2026-09-08 |
| #4 | Fix CPU-fallback path to rebuild Verlet lists only past the skin (48 s → 35 s on 32k) | 2026-09-08 |
| #5 | Port native DNA/DNA2 force+torque kernel faithfully from `CUDA_DNA.cuh` (HB, cross/coaxial stacking, Debye-Hückel, FENE cap, body-frame torque); skin-aware rebuilds. Native 32k run: crash → 0.79 s, energies match CPU | 2026-09-09 |
| #6 | Batch the whole native MD step into one GPU submission (5 waits/step → 1). 32k/2000 steps wall: 30.6 s CPU → 3.3 s Metal-native | 2026-09-09 |

---

## Session Log

### 2026-09-08
- Started: user asked to keep `metal` up to date with master, build the Metal backend, verify, optimize, test on examples.
- Progress: merged master (4 conflicts), installed Metal toolchain, built all 3 binaries + shaders.metallib. Found the CPU-fallback path correct (~1e-6 vs CPU) but slower than CPU; native GPU kernels produced wrong forces and blew up. Fixed the fallback list-rebuild frequency. Backups: branch `metal-backup-2026-09-08`, tag `metal-pre-merge-2026-09-08`.
- Stopped at: native kernel needs a real port.

### 2026-09-09
- Started: stabilize the native GPU DNA kernel.
- Progress: rewrote `src/Metal/Shaders/dna_kernels.metal` as a faithful port of `src/CUDA/Interactions/CUDA_DNA.cuh` — added hydrogen bonding, cross stacking, coaxial stacking (oxDNA1+2), Debye-Hückel, the max-backbone-force FENE cap, correct `_f1D`/`_f2D`/`_f4`/`_f4D`, and the body-frame torque transform. Native DNA/DNA2 now match the CPU backend to ~1e-6 / ~1e-4 and the 32k example runs stably. Added `MetalBaseList::lists_are_old()` (host displacement check) so the native path stops rebuilding Verlet lists every step. Then batched the entire MD step (first step → zero → forces → second step → thermostat) into one command buffer / one host sync: 32k×2000 steps wall time 30.6 s (CPU backend) → 3.3 s (Metal native), ~9×. Commits `77bc1ee1`, `580660ad` pushed to `origin/metal`.
- Next: user wants three precision tiers (`float`, `hardmixed` = GPU-float/CPU-double, `mixed` = df64 two-float), OpenMP for CPU MD/MC, full energy validation + speed-up report, docs. Metal has NO GPU double (`double` is a compile error) — `mixed` must be double-float emulation.
- Stopped at: about to start Phase A (precision-mode scaffolding).
