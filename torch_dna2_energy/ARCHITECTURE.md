# Torch DNA2 Energy Calculator Architecture

## Overview

This document presents a comprehensive architectural design for a PyTorch C++ implementation of the DNA2 energy calculator from oxDNA. The design leverages PyTorch's C++ frontend (libtorch) to provide GPU acceleration and automatic differentiation while maintaining mathematical accuracy and compatibility with the original oxDNA implementation.

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DNA2EnergyCalculator                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Energy Core   │  │  Force Core     │  │  Gradient Core  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Interaction     │  │ Mathematical    │  │ Mesh            │ │
│  │ Manager         │  │ Functions       │  │ Interpolation   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Particle        │  │ Parameter       │  │ Tensor          │ │
│  │ Manager         │  │ Manager         │  │ Operations      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    PyTorch Backend                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   CPU Device    │  │   GPU Device    │  │ Memory Manager  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Class Hierarchy

```cpp
// Core calculator class
class DNA2EnergyCalculator {
    // Main interface for energy/force calculations
};

// Interaction management
class InteractionManager {
    // Manages different interaction types
};

class BaseInteraction {
    // Abstract base for all interactions
};

class BackboneInteraction : public BaseInteraction { };
class StackingInteraction : public BaseInteraction { }
class HydrogenBondingInteraction : public BaseInteraction { }
class ExcludedVolumeInteraction : public BaseInteraction { }
class CrossStackingInteraction : public BaseInteraction { }
class CoaxialStackingInteraction : public BaseInteraction { }
class DebyeHuckelInteraction : public BaseInteraction { }

// Mathematical functions
class MathematicalFunctions {
    // f1, f2, f4, f5 functions and derivatives
};

// Mesh interpolation
class MeshInterpolator {
    // Angular potential interpolation
};

// Particle and parameter management
class ParticleManager {
    // Particle data and operations
};

class ParameterManager {
    // Model parameters and constants
};
```

## 2. Data Structures

### 2.1 Particle Representation

```cpp
struct DNAParticle {
    // Position and orientation tensors
    torch::Tensor positions;        // [N, 3] - particle positions
    torch::Tensor orientations;     // [N, 3, 3] - rotation matrices
    torch::Tensor orientations_t;   // [N, 3, 3] - transpose of orientations
    
    // Interaction centers
    torch::Tensor backbone_centers; // [N, 3] - backbone positions
    torch::Tensor stack_centers;    // [N, 3] - stacking positions
    torch::Tensor base_centers;     // [N, 3] - base positions
    
    // Particle properties
    torch::Tensor types;            // [N] - particle types (0-4)
    torch::Tensor btypes;           // [N] - base types for pairing
    torch::Tensor strand_ids;       // [N] - strand identifiers
    
    // Bonding information
    torch::Tensor n3_neighbors;     // [N] - 3' neighbor indices
    torch::Tensor n5_neighbors;     // [N] - 5' neighbor indices
    torch::Tensor bonded_mask;      // [N, N] - bonding matrix
};
```

### 2.2 Parameter Storage

```cpp
struct DNA2Parameters {
    // FENE parameters
    float fene_eps = 2.0f;
    float fene_r0 = 0.7564f;  // oxDNA2 specific
    float fene_delta = 0.25f;
    
    // Excluded volume parameters
    float excl_eps = 2.0f;
    torch::Tensor excl_sigma;    // [4] - sigma for different site types
    torch::Tensor excl_rstar;    // [4] - rstar for different site types
    torch::Tensor excl_b;        // [4] - b parameter for different site types
    torch::Tensor excl_rc;       // [4] - cutoff for different site types
    
    // Hydrogen bonding parameters
    float hydr_eps = 1.0678f;    // oxDNA2 specific
    float hydr_a = 8.0f;
    float hydr_rc = 0.75f;
    float hydr_r0 = 0.4f;
    torch::Tensor hydr_eps_matrix; // [5, 5] - sequence-dependent epsilons
    
    // Stacking parameters
    float stck_base_eps = 1.3523f;  // oxDNA2 specific
    float stck_fact_eps = 2.6717f;  // oxDNA2 specific
    float stck_a = 6.0f;
    float stck_rc = 0.9f;
    float stck_r0 = 0.4f;
    torch::Tensor stck_eps_matrix; // [5, 5] - sequence-dependent epsilons
    
    // Cross-stacking parameters
    float crst_k = 47.5f;
    float crst_r0 = 0.575f;
    float crst_rc = 0.675f;
    
    // Coaxial stacking parameters
    float cxst_k = 58.5f;  // oxDNA2 specific
    float cxst_r0 = 0.4f;
    float cxst_rc = 0.6f;
    
    // Debye-Huckel parameters
    float salt_concentration = 0.5f;
    float dh_lambda = 0.3616455f;
    float dh_strength = 0.0543f;
    bool dh_half_charged_ends = true;
    
    // Temperature
    float temperature = 0.1f;  // oxDNA reduced units
    
    // Grooving
    bool grooving = true;
};
```

### 2.3 Interaction Result Structure

```cpp
struct InteractionResult {
    torch::Tensor total_energy;    // [1] or [batch_size]
    torch::Tensor forces;          // [N, 3] or [batch_size, N, 3]
    torch::Tensor torques;         // [N, 3] or [batch_size, N, 3]
    
    // Energy breakdown by interaction type
    torch::Tensor backbone_energy;
    torch::Tensor stacking_energy;
    torch::Tensor hb_energy;
    torch::Tensor excl_energy;
    torch::Tensor crst_energy;
    torch::Tensor cxst_energy;
    torch::Tensor dh_energy;
};
```

## 3. Mathematical Functions Implementation

### 3.1 Function Architecture

```cpp
class MathematicalFunctions {
public:
    // Radial functions
    static torch::Tensor f1(const torch::Tensor& r, int type, 
                           const torch::Tensor& n3_types, 
                           const torch::Tensor& n5_types,
                           const DNA2Parameters& params);
    
    static torch::Tensor f1_derivative(const torch::Tensor& r, int type,
                                      const torch::Tensor& n3_types,
                                      const torch::Tensor& n5_types,
                                      const DNA2Parameters& params);
    
    static torch::Tensor f2(const torch::Tensor& r, int type,
                           const DNA2Parameters& params);
    
    static torch::Tensor f2_derivative(const torch::Tensor& r, int type,
                                      const DNA2Parameters& params);
    
    // Angular functions
    static torch::Tensor f4(const torch::Tensor& theta, int type,
                           const DNA2Parameters& params);
    
    static torch::Tensor f4_derivative(const torch::Tensor& theta, int type,
                                      const DNA2Parameters& params);
    
    static torch::Tensor f4_derivative_sin(const torch::Tensor& theta, int type,
                                          const DNA2Parameters& params);
    
    // Phi functions
    static torch::Tensor f5(const torch::Tensor& phi, int type,
                           const DNA2Parameters& params);
    
    static torch::Tensor f5_derivative(const torch::Tensor& phi, int type,
                                      const DNA2Parameters& params);
    
private:
    // Helper functions for piecewise definitions
    static torch::Tensor f1_core(const torch::Tensor& r, int type,
                                const torch::Tensor& eps,
                                const DNA2Parameters& params);
    
    static torch::Tensor f2_core(const torch::Tensor& r, int type,
                                const DNA2Parameters& params);
    
    static torch::Tensor f4_core(const torch::Tensor& theta, int type,
                                const DNA2Parameters& params);
};
```

### 3.2 Vectorized Implementation Strategy

All mathematical functions will be implemented using PyTorch's vectorized operations to maximize GPU performance:

```cpp
// Example: f1 function implementation
torch::Tensor MathematicalFunctions::f1(const torch::Tensor& r, int type,
                                       const torch::Tensor& n3_types,
                                       const torch::Tensor& n5_types,
                                       const DNA2Parameters& params) {
    // Vectorized piecewise function implementation
    auto eps = params.f1_eps[type].index({n3_types, n5_types});
    auto shift = params.f1_shift[type].index({n3_types, n5_types});
    
    // Create masks for different regions
    auto mask_low = r < params.f1_rlow[type];
    auto mask_mid = (r >= params.f1_rlow[type]) & (r <= params.f1_rhigh[type]);
    auto mask_high = r > params.f1_rhigh[type];
    
    // Compute values for each region
    auto val_low = params.f1_blow[type] * torch::square(r - params.f1_rc_low[type]);
    auto val_mid = torch::square(1 - torch::exp(-(r - params.f1_r0[type]) * params.f1_a[type]));
    auto val_high = params.f1_bhigh[type] * torch::square(r - params.f1_rc_high[type]);
    
    // Combine results
    auto result = torch::zeros_like(r);
    result = torch::where(mask_low, eps * val_low, result);
    result = torch::where(mask_mid, eps * val_mid - shift, result);
    result = torch::where(mask_high, eps * val_high, result);
    
    return result;
}
```

## 4. Mesh Interpolation System

### 4.1 Mesh Data Structure

```cpp
class MeshInterpolator {
private:
    struct MeshData {
        torch::Tensor x_values;      // [N_points] - cosine values
        torch::Tensor y_values;      // [N_points] - function values
        torch::Tensor y_derivatives; // [N_points] - derivative values
        int mesh_type;
    };
    
    std::unordered_map<int, MeshData> meshes_;
    
public:
    void initialize_meshes(const DNA2Parameters& params);
    
    torch::Tensor interpolate(int mesh_type, const torch::Tensor& cos_values);
    torch::Tensor interpolate_derivative(int mesh_type, const torch::Tensor& cos_values);
    
private:
    torch::Tensor cubic_hermite_interpolate(const torch::Tensor& x,
                                          const torch::Tensor& y,
                                          const torch::Tensor& dy,
                                          const torch::Tensor& query_points);
};
```

### 4.2 Mesh Initialization

```cpp
void MeshInterpolator::initialize_meshes(const DNA2Parameters& params) {
    // Initialize all 13 mesh types for DNA2
    initialize_mesh(HYDR_F4_THETA1, params, HYDR_T1_MESH_POINTS, 
                   HYDR_THETA1_T0, HYDR_THETA1_TC, HYDR_THETA1_TS);
    // ... initialize other meshes
    
    // Special handling for CXST_F4_THETA1 with pure harmonic addition
    initialize_cxst_theta1_mesh(params);
}
```

## 5. Energy Calculation Pipeline

### 5.1 Pipeline Organization

```cpp
class EnergyCalculationPipeline {
public:
    InteractionResult compute_energy(const DNAParticle& particles,
                                   const DNA2Parameters& params,
                                   bool compute_forces = true,
                                   bool compute_torques = true);
    
private:
    // Distance calculations
    torch::Tensor compute_pairwise_distances(const DNAParticle& particles);
    torch::Tensor compute_interaction_center_distances(const DNAParticle& particles);
    
    // Interaction calculations
    torch::Tensor compute_backbone_energy(const DNAParticle& particles,
                                        const torch::Tensor& distances,
                                        const DNA2Parameters& params);
    
    torch::Tensor compute_stacking_energy(const DNAParticle& particles,
                                        const torch::Tensor& distances,
                                        const DNA2Parameters& params);
    
    torch::Tensor compute_hb_energy(const DNAParticle& particles,
                                  const torch::Tensor& distances,
                                  const DNA2Parameters& params);
    
    // Force and torque calculations
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params);
};
```

### 5.2 Batch Processing Strategy

```cpp
class BatchProcessor {
public:
    InteractionResult process_batch(const std::vector<DNAParticle>& batch_particles,
                                  const DNA2Parameters& params);
    
private:
    DNAParticle stack_particles(const std::vector<DNAParticle>& batch);
    std::vector<InteractionResult> unstack_results(const InteractionResult& batch_result,
                                                  int batch_size);
};
```

## 6. Torch Tensor Integration Strategy

### 6.1 Device Management

```cpp
class DeviceManager {
public:
    static DeviceManager& instance();
    
    void set_device(torch::Device device);
    torch::Device get_device() const { return device_; }
    
    bool is_cuda() const { return device_.is_cuda(); }
    
private:
    torch::Device device_{torch::kCPU};
};
```

### 6.2 Tensor Operations

```cpp
class TensorOperations {
public:
    // Vector operations
    static torch::Tensor cross_product(const torch::Tensor& a, const torch::Tensor& b);
    static torch::Tensor dot_product(const torch::Tensor& a, const torch::Tensor& b);
    static torch::Tensor normalize(const torch::Tensor& v);
    
    // Matrix operations
    static torch::Tensor matrix_transpose(const torch::Tensor& m);
    static torch::Tensor matrix_multiply(const torch::Tensor& a, const torch::Tensor& b);
    
    // Geometric calculations
    static torch::Tensor compute_angles(const torch::Tensor& v1, const torch::Tensor& v2);
    static torch::Tensor compute_dihedrals(const torch::Tensor& v1, const torch::Tensor& v2,
                                          const torch::Tensor& v3, const torch::Tensor& v4);
    
    // Distance calculations with periodic boundaries
    static torch::Tensor minimum_image_distance(const torch::Tensor& r1, const torch::Tensor& r2,
                                               const torch::Tensor& box_size);
};
```

## 7. Memory Management and GPU Acceleration

### 7.1 Memory Pool Management

```cpp
class MemoryPool {
private:
    std::unordered_map<std::string, torch::Tensor> tensor_cache_;
    size_t max_cache_size_;
    
public:
    torch::Tensor get_tensor(const std::string& key, const torch::TensorOptions& options,
                           const std::vector<int64_t>& shape);
    
    void clear_cache();
    void set_max_cache_size(size_t size) { max_cache_size_ = size; }
};
```

### 7.2 GPU Optimization Strategies

1. **Kernel Fusion**: Combine multiple operations into single CUDA kernels
2. **Memory Coalescing**: Ensure memory access patterns are optimized
3. **Shared Memory**: Use shared memory for frequently accessed data
4. **Batch Processing**: Process multiple configurations simultaneously
5. **Asynchronous Operations**: Use CUDA streams for overlapping computation

### 7.3 Automatic Differentiation Support

```cpp
class AutoDiffSupport {
public:
    static torch::Tensor compute_energy_with_grad(const DNAParticle& particles,
                                                 const DNA2Parameters& params);
    
    static std::tuple<torch::Tensor, torch::Tensor> compute_energy_and_forces(
        const DNAParticle& particles,
        const DNA2Parameters& params);
    
private:
    static torch::Tensor energy_function(const torch::Tensor& positions,
                                       const torch::Tensor& orientations,
                                       const DNA2Parameters& params);
};
```

## 8. Interface Design

### 8.1 Main Interface Class

```cpp
class DNA2EnergyCalculator {
public:
    // Constructor
    DNA2EnergyCalculator(const DNA2Parameters& params = DNA2Parameters{},
                        torch::Device device = torch::kCPU);
    
    // Single configuration calculations
    torch::Tensor compute_energy(const DNAParticle& particles);
    torch::Tensor compute_forces(const DNAParticle& particles);
    std::tuple<torch::Tensor, torch::Tensor> compute_energy_and_forces(
        const DNAParticle& particles);
    
    // Batch processing
    torch::Tensor compute_energy_batch(const std::vector<DNAParticle>& batch);
    torch::Tensor compute_forces_batch(const std::vector<DNAParticle>& batch);
    
    // Gradient-based optimization support
    torch::Tensor compute_energy_autograd(const torch::Tensor& positions,
                                        const torch::Tensor& orientations);
    
    // Parameter management
    void set_parameters(const DNA2Parameters& params);
    const DNA2Parameters& get_parameters() const { return params_; }
    
    // Device management
    void to(torch::Device device);
    torch::Device device() const { return device_manager_.get_device(); }
    
private:
    DNA2Parameters params_;
    DeviceManager device_manager_;
    EnergyCalculationPipeline pipeline_;
    BatchProcessor batch_processor_;
    MemoryPool memory_pool_;
};
```

### 8.2 Convenience Functions

```cpp
// Factory functions
DNA2EnergyCalculator create_default_calculator();
DNA2EnergyCalculator create_gpu_calculator(int gpu_id = 0);

// Conversion utilities
DNAParticle from_oxdna_format(const std::vector<BaseParticle*>& oxdna_particles);
std::vector<BaseParticle*> to_oxdna_format(const DNAParticle& torch_particles);

// I/O utilities
void save_configuration(const DNAParticle& particles, const std::string& filename);
DNAParticle load_configuration(const std::string& filename);
```

## 9. Performance Considerations

### 9.1 Optimization Strategies

1. **Vectorization**: All operations implemented using PyTorch's vectorized functions
2. **Memory Layout**: Optimized tensor layouts for GPU memory access patterns
3. **Caching**: Cache frequently computed values (e.g., interaction centers)
4. **Parallelization**: Leverage PyTorch's built-in parallelization
5. **Precision**: Support for both float32 and float64 precision

### 9.2 Benchmarking and Profiling

```cpp
class PerformanceProfiler {
public:
    void start_timing(const std::string& operation);
    void end_timing(const std::string& operation);
    void print_report() const;
    
private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::unordered_map<std::string, double> accumulated_times_;
};
```

## 10. Error Handling and Validation

### 10.1 Input Validation

```cpp
class InputValidator {
public:
    static void validate_particles(const DNAParticle& particles);
    static void validate_parameters(const DNA2Parameters& params);
    static void validate_device_compatibility(const torch::Device& device);
    
private:
    static void check_tensor_shapes(const torch::Tensor& tensor,
                                   const std::vector<int64_t>& expected_shape);
    static void check_tensor_device(const torch::Tensor& tensor,
                                   const torch::Device& expected_device);
};
```

### 10.2 Exception Handling

```cpp
class DNA2EnergyException : public std::exception {
private:
    std::string message_;
public:
    explicit DNA2EnergyException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
};
```

## 11. Testing Strategy

### 11.1 Unit Tests

- Mathematical function accuracy tests
- Mesh interpolation validation
- Individual interaction term tests
- Gradient verification against numerical derivatives

### 11.2 Integration Tests

- End-to-end energy calculation validation against oxDNA
- Force calculation accuracy tests
- Batch processing consistency tests
- GPU vs CPU numerical accuracy verification

### 11.3 Performance Tests

- Scaling with particle number
- GPU vs CPU performance comparison
- Memory usage profiling
- Batch size optimization

## 12. Implementation Roadmap

### Phase 1: Core Infrastructure
1. Basic class hierarchy
2. Data structures
3. Mathematical functions
4. Device management

### Phase 2: Interaction Implementation
1. Individual interaction terms
2. Mesh interpolation system
3. Energy calculation pipeline
4. Force and torque calculations

### Phase 3: Advanced Features
1. Batch processing
2. Automatic differentiation
3. GPU optimization
4. Performance tuning

### Phase 4: Integration and Testing
1. Comprehensive testing suite
2. Documentation
3. Example applications
4. Performance benchmarking

## 13. Conclusion

This architecture provides a comprehensive foundation for implementing the DNA2 energy calculator in PyTorch C++. The design emphasizes:

- **Performance**: Leveraging GPU acceleration and vectorized operations
- **Accuracy**: Maintaining mathematical fidelity to the original oxDNA implementation
- **Flexibility**: Supporting both CPU and GPU execution, batch processing, and automatic differentiation
- **Usability**: Providing a clean, intuitive interface for easy integration
- **Maintainability**: Modular design with clear separation of concerns

The architecture is designed to be extensible, allowing for future enhancements such as additional interaction types, advanced optimization techniques, and integration with machine learning workflows.