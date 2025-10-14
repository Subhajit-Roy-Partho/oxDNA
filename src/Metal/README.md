# Metal GPU Backend for oxDNA

This directory contains the Metal GPU implementation for oxDNA, designed to run on Apple Silicon (M-series) GPUs including the M4.

## Overview

The Metal backend provides GPU acceleration for oxDNA molecular dynamics simulations using Apple's Metal API. It mirrors the structure and functionality of the CUDA backend but is optimized for Apple's unified memory architecture and GPU compute capabilities.

## Directory Structure

```
src/Metal/
├── metal_defs.h              # Core Metal definitions and types
├── MetalUtils.h/.mm          # Utility functions for Metal operations
├── Backends/                 # Simulation backends
│   ├── MetalBaseBackend.h/.mm    # Base Metal backend
│   └── MD_MetalBackend.h/.mm     # Molecular dynamics backend
├── Interactions/             # Force field implementations (TODO)
├── Lists/                    # Neighbor list implementations (TODO)
├── Thermostats/             # Temperature control (TODO)
├── Shaders/                 # Metal shader kernels
│   ├── common.metal         # Common shader functions
│   └── md_kernels.metal     # MD-specific kernels
└── metal_utils/             # Utility structures
    └── MetalBox.h           # Simulation box structure
```

## Key Features

### Implemented
- ✅ Base Metal backend infrastructure
- ✅ Device management and initialization
- ✅ Memory management with Metal buffers
- ✅ Velocity Verlet integration kernels
- ✅ Position and velocity updates with PBC
- ✅ Kernel pipeline management
- ✅ Host-GPU data transfer

### TODO
- ⬜ Force calculation kernels (DNA, RNA, LJ, etc.)
- ⬜ Neighbor list implementations
- ⬜ Thermostat implementations (Brownian, Langevin, Bussi)
- ⬜ Barostat for NPT ensemble
- ⬜ Stress tensor calculation
- ⬜ External forces
- ⬜ Particle sorting optimization
- ⬜ Rigid body dynamics
- ⬜ Performance optimizations

## Requirements

- macOS 10.15 or later
- Apple Silicon (M1/M2/M3/M4) or Metal-capable AMD/Intel GPU
- Xcode command line tools
- CMake 3.5 or later

## Building

To build oxDNA with Metal support:

```bash
mkdir build
cd build
cmake -DMETAL=ON ..
make
```

Optional flags:
- `-DMETAL_DOUBLE=ON` - Use double precision (default: single precision)
- `-DCMAKE_BUILD_TYPE=Debug` - Build with debug symbols

## Usage

Configure your input file to use the Metal backend:

```
backend = MD_Metal
backend_precision = float  # or 'double' if compiled with METAL_DOUBLE

# Metal-specific options
Metal_sort_every = 0              # Particle sorting interval (0 = disabled)
threads_per_threadgroup = 256     # Threads per threadgroup (default: 256)
Metal_avoid_cpu_calculations = 1  # Avoid CPU fallback computations
```

## Architecture

### Memory Model
Metal uses a unified memory architecture on Apple Silicon, which means:
- Host and device share the same physical memory
- Data transfers are fast (pointer sharing in many cases)
- Uses `MTLResourceStorageModeShared` for CPU-GPU accessible buffers

### Compute Model
- **Threadgroups**: Equivalent to CUDA blocks
- **Threads**: Individual execution units
- **SIMD width**: 32 on Apple GPUs (similar to CUDA warps)

### Type System
- `m_number`: Configurable precision (float/double)
- `m_number3`: 3D vector (simd_float3/simd_double3)
- `m_number4`: 4D vector (simd_float4/simd_double4)
- `MetalBonds`: Bond connectivity structure
- `MetalBox`: Simulation box with PBC

## Implementation Details

### Kernel Execution
Kernels are dispatched using Metal compute command encoders:
1. Create command buffer
2. Create compute command encoder
3. Set pipeline state and buffers
4. Dispatch threads
5. End encoding and commit
6. Wait for completion (or async)

### Velocity Verlet Integration
The MD timestep follows the standard velocity Verlet scheme:
1. **First step**: Update velocities (half-step) and positions
2. **Force calculation**: Compute forces on new positions
3. **Second step**: Update velocities (second half-step)

### Periodic Boundary Conditions
Implemented in shaders using minimum image convention:
```metal
r -= box_sides * rint(r * inv_sides)
```

## Performance Considerations

- **Threadgroup size**: Default 256, optimal for Apple GPUs
- **Memory coalescing**: Metal handles this automatically with unified memory
- **Buffer sharing**: Minimize host-device copies using shared buffers
- **Asynchronous execution**: Can overlap compute and data transfers

## Comparison with CUDA

| Feature | CUDA | Metal |
|---------|------|-------|
| Memory model | Separate host/device | Unified (on Apple Silicon) |
| Thread organization | Grid/Block/Thread | Grid/Threadgroup/Thread |
| SIMD width | 32 (warp) | 32 (SIMD group) |
| Language | CUDA C++ | Metal Shading Language |
| Portability | NVIDIA GPUs | Apple devices |

## Contributing

When adding new features:
1. Follow the existing CUDA structure for consistency
2. Add corresponding shader implementations in `Shaders/`
3. Update pipeline creation in backend initialization
4. Test on both M-series and Intel Macs if possible

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/best_practices)

## License

This Metal implementation follows the same license as oxDNA.
