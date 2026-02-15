# oxDNA Repository Map (CPU/CUDA/Metal)

## Core Simulation Flow
- `src/Managers/SimManager.cpp`: top-level simulation loop, step accounting, output scheduling.
- `src/Backends/SimBackend.cpp`: backend-agnostic initialization flow (particles, lists, interaction, observables).
- `src/Backends/MDBackend.cpp`: shared MD logic used by concrete backends.
- `src/Backends/MD_CPUBackend.cpp`: CPU MD stepping, integration, thermostat, force invocation.
- `src/Backends/BackendFactory.cpp`: backend selection (`CPU`, `CUDA`, `Metal`, etc.).

## CPU Interactions
- `src/Interactions/InteractionFactory.cpp`: CPU interaction type dispatch (`DNA`, `DNA2`, `LJ`, `RNA`, `RNA2`, `patchy`, `TEP`, ...).
- `src/Interactions/BaseInteraction.{h,cpp}`: interaction interface, topology base handling, energy split support.
- `src/Interactions/DNAInteraction.cpp`: DNA/DNA2 forcefield implementation (bonded + nonbonded).
- `src/Interactions/RNAInteraction.cpp`: RNA/RNA2 forcefield implementation.
- `src/Interactions/LJInteraction.cpp`: Lennard-Jones model.
- `src/Interactions/PatchyInteraction.cpp`: patchy particle interaction.
- `src/Interactions/TEPInteraction.cpp`: TEP model.

## CUDA Path
- `src/CUDA/Backends/MD_CUDABackend.{h,cu}`: CUDA MD integration and force orchestration.
- `src/CUDA/Interactions/CUDAInteractionFactory.cu`: CUDA interaction dispatch.
- `src/CUDA/Interactions/CUDADNAInteraction.{h,cu}`: CUDA DNA/DNA2 interaction path.
- `src/CUDA/Interactions/CUDALJInteraction.{h,cu}`: CUDA LJ.
- `src/CUDA/Interactions/CUDARNAInteraction.{h,cu}`: CUDA RNA/RNA2.
- `src/CUDA/Interactions/CUDAPatchyInteraction.{h,cu}`: CUDA patchy.
- `src/CUDA/Interactions/CUDATEPInteraction.{h,cu}`: CUDA TEP.

## Metal Path
- `src/Metal/MetalBackendFactory.mm`: Metal backend entry point registration.
- `src/Metal/Backends/MetalBaseBackend.mm`: Metal device init, shader library load, host/device base buffers.
- `src/Metal/Backends/MD_MetalBackend.mm`: Metal MD stepping and integration orchestration.
- `src/Metal/Lists/MetalListFactory.mm`: Metal list type selection.
- `src/Metal/Interactions/MetalInteractionFactory.mm`: Metal interaction dispatch.
- `src/Metal/Interactions/MetalBaseInteraction.{h,mm}`: common Metal interaction settings (including fallback mode).
- `src/Metal/Interactions/MetalDNAInteraction.mm`: native DNA kernel path + CPU fallback path.
- `src/Metal/Interactions/MetalLJInteraction.mm`: LJ adapter (CPU-force fallback).
- `src/Metal/Interactions/MetalRNAInteraction.mm`: RNA/RNA2 adapter (CPU-force fallback).
- `src/Metal/Interactions/MetalPatchyInteraction.mm`: patchy adapter (CPU-force fallback).
- `src/Metal/Interactions/MetalTEPInteraction.mm`: TEP adapter (CPU-force fallback).
- `src/Metal/Interactions/MetalCPUForceFallback.mm`: shared CPU force evaluation helper for Metal adapters.
- `src/Metal/Thermostats/MetalThermostatFactory.mm`: Metal thermostat dispatch (`brownian`, `john`, `no`).

## Metal Shaders and Build Hooks
- `src/Metal/Shaders/md_kernels.metal`: MD integration kernels and force/torque zeroing kernels.
- `src/Metal/Shaders/dna_kernels.metal`: DNA force kernels.
- `src/Metal/Shaders/list_kernels.metal`: neighbor list kernels.
- `src/Metal/Shaders/thermostat_kernels.metal`: thermostat kernels.
- `src/Metal/CMakeLists.txt`: Metal source list, shader compilation (`metal` + `metallib`), install/copy steps.
- `CMakeLists.txt`: top-level target wiring.

## Validation and Tests
- `comparison_run/run_comparison.py`: legacy CPU-vs-Metal comparison helper.
- `comparison_run/validate_metal_forcefields.py`: deterministic parity validator across DNA/DNA2/LJ/RNA/RNA2/patchy/TEP.
- `test/METAL/SIMPLE_MD/`: base Metal DNA fixture.
- `test/DNA/FORCE_FIELD/`: DNA2 forcefield fixture data.
- `test/RNA/FORCE_FIELD/`: RNA/RNA2 fixture data.
- `test/LJ/`: LJ fixture data.

## Checklist: Adding a New Forcefield to Metal
1. Add CPU interaction support first in `src/Interactions/` and wire `src/Interactions/InteractionFactory.cpp`.
2. Add CUDA implementation (if parity target includes CUDA) and wire `src/CUDA/Interactions/CUDAInteractionFactory.cu`.
3. Add Metal adapter in `src/Metal/Interactions/` and wire `src/Metal/Interactions/MetalInteractionFactory.mm`.
4. Decide fallback behavior:
   - default correctness mode via `MetalCPUForceFallback`, or
   - native Metal kernels under `Metal_avoid_cpu_calculations = 1`.
5. If native kernels are added, update shader files in `src/Metal/Shaders/` and confirm pipeline creation in Metal backend/interaction code.
6. Update `src/Metal/CMakeLists.txt` with any new `.mm` source and shader compilation needs.
7. Extend `comparison_run/validate_metal_forcefields.py` with a deterministic scenario and parity threshold check.
8. Run validation and ensure:
   - no `ERROR` markers in logs,
   - no NaN/Inf energies,
   - CPU-vs-Metal relative potential error within target tolerance.
