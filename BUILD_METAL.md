# Metal Backend Build Instructions

## ✅ Build Status: SUCCESS

The oxDNA Metal backend for Apple Silicon (M-series) GPUs has been successfully compiled!

## Build Summary

### Configuration
- **Build System**: CMake + Ninja
- **GPU Backend**: Metal (Apple GPU API)
- **Precision**: Single precision (float)
- **Build Type**: Release
- **Platform**: macOS (Apple Silicon M4 compatible)

### Compiled Executables

Located in `build_metal/bin/`:

- `oxDNA` (3.8 MB) - Main simulation executable
- `DNAnalysis` (3.5 MB) - Analysis tool
- `confGenerator` (3.1 MB) - Configuration generator

### Metal Symbols Verified

The build includes Metal-specific symbols:
- `create_metal_backend()` - Factory function
- `MD_MetalBackend` - Molecular dynamics backend
- `MetalUtils` - GPU utility functions
- Metal buffer allocation and management functions

## How to Build

### Prerequisites

1. macOS 10.15 or later
2. Apple Silicon (M1/M2/M3/M4) or Metal-capable GPU
3. Xcode Command Line Tools
4. CMake 3.5 or later
5. Ninja build system

```bash
# Install prerequisites (if not already installed)
xcode-select --install
brew install cmake ninja
```

### Build Commands

```bash
# Create build directory
mkdir build_metal
cd build_metal

# Configure with CMake
cmake -G Ninja -DMETAL=ON -DCMAKE_BUILD_TYPE=Release ..

# Compile
ninja -j8

# Executables will be in build_metal/bin/
```

### Build Options

- `-DMETAL=ON` - Enable Metal GPU support
- `-DMETAL_DOUBLE=ON` - Use double precision (default: single/float)
- `-DCMAKE_BUILD_TYPE=Release` - Optimized build
- `-DCMAKE_BUILD_TYPE=Debug` - Debug build with symbols

## Using the Metal Backend

### Input File Configuration

To use the Metal backend, add these options to your input file:

```
backend = Metal
backend_precision = float  # or 'double' if compiled with METAL_DOUBLE=ON

# Metal-specific options
Metal_sort_every = 0              # Particle sorting interval (0 = disabled)
threads_per_threadgroup = 256     # Threads per threadgroup
Metal_avoid_cpu_calculations = 1  # Avoid CPU fallback
```

### Example Input File

```
backend = Metal
sim_type = MD

# System parameters
backend_precision = float
T = 300K
dt = 0.005

# Metal GPU settings
threads_per_threadgroup = 256
Metal_sort_every = 0

# Other standard oxDNA options...
```

## Implementation Status

### ✅ Completed Features

- Metal device management and initialization
- Memory allocation with Metal buffers (unified memory)
- Velocity Verlet integration kernels
- Position and velocity updates with periodic boundary conditions
- Kernel pipeline management
- Host-GPU data transfer
- Backend factory integration

### ⏳ TODO Features

The following features are planned but not yet implemented:

- Force calculation kernels (DNA, RNA, LJ interactions)
- Neighbor list implementations
- Thermostats (Brownian, Langevin, Bussi)
- Barostat for NPT ensemble
- Stress tensor calculation
- External forces
- Particle sorting optimization (Hilbert curve)
- Rigid body dynamics
- Full performance optimizations

## Performance Notes

### Apple Silicon Advantages

- **Unified Memory Architecture**: Host and GPU share physical memory
- **Fast Data Transfer**: No explicit copying in many cases
- **SIMD Width**: 32 threads (similar to CUDA warps)
- **Optimal Threadgroup Size**: 256 threads per threadgroup

### Memory Model

Metal uses `MTLResourceStorageModeShared` for CPU-GPU accessible buffers, which is very efficient on Apple Silicon's unified memory architecture.

## Files Created

### Core Files

```
src/Metal/
├── metal_defs.h                 # Metal types and definitions
├── MetalUtils.h/mm             # Utility functions
├── MetalBackendFactory.mm      # Backend factory
├── CMakeLists.txt              # Build configuration
├── README.md                   # Documentation
├── Backends/
│   ├── MetalBaseBackend.h/mm   # Base backend
│   └── MD_MetalBackend.h/mm    # MD backend
├── Shaders/
│   ├── common.metal            # Common shader functions
│   └── md_kernels.metal        # MD kernels
└── metal_utils/
    └── MetalBox.h              # Simulation box
```

### Modified Files

- `CMakeLists.txt` - Added Metal option
- `src/CMakeLists.txt` - Metal source integration
- `src/Backends/BackendFactory.cpp` - Metal backend factory

## Troubleshooting

### Common Issues

1. **Metal not found**: Ensure you're on macOS 10.15+ with Metal support
2. **Objective-C++ errors**: Make sure `.mm` files are compiled with Objective-C++
3. **Linking errors**: Verify Metal frameworks are linked (Metal, Foundation, CoreGraphics)

### Verification

Check if Metal backend is available:

```bash
# Binary should contain Metal symbols
nm bin/oxDNA | grep -i metal

# Should see symbols like:
# - create_metal_backend
# - MD_MetalBackend
# - MetalUtils
```

## Next Steps

To complete the Metal backend implementation:

1. **Implement Force Kernels**: Port DNA/RNA/LJ interaction calculations
2. **Add Neighbor Lists**: Implement Verlet lists for Metal
3. **Thermostats**: Add temperature control methods
4. **Optimize**: Profile and optimize for Apple GPU architecture
5. **Test**: Validate against CPU and CUDA results

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices](https://developer.apple.com/documentation/metal/best_practices)
- [oxDNA Documentation](https://lorenzo-rovigatti.github.io/oxDNA/)

## Contributors

Metal backend implementation for Apple Silicon M4 GPU.
