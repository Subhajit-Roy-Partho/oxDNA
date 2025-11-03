#pragma once

#include <torch/torch.h>
#include "dna2_data_structures.h"

namespace dna2 {

/**
 * @brief Mathematical functions for DNA2 energy calculations
 * 
 * Implements the f1, f2, f4, f5 functions and their derivatives
 * as used in the oxDNA2 model.
 */
class MathematicalFunctions {
public:
    // Radial functions
    static torch::Tensor f1(const torch::Tensor& r, FunctionType type, 
                           const torch::Tensor& n3_types, 
                           const torch::Tensor& n5_types,
                           const DNA2Parameters& params);
    
    static torch::Tensor f1_derivative(const torch::Tensor& r, FunctionType type,
                                      const torch::Tensor& n3_types,
                                      const torch::Tensor& n5_types,
                                      const DNA2Parameters& params);
    
    static torch::Tensor f2(const torch::Tensor& r, FunctionType type,
                           const DNA2Parameters& params);
    
    static torch::Tensor f2_derivative(const torch::Tensor& r, FunctionType type,
                                      const DNA2Parameters& params);
    
    // Angular functions
    static torch::Tensor f4(const torch::Tensor& theta, FunctionType type,
                           const DNA2Parameters& params);
    
    static torch::Tensor f4_derivative(const torch::Tensor& theta, FunctionType type,
                                      const DNA2Parameters& params);
    
    static torch::Tensor f4_derivative_sin(const torch::Tensor& theta, FunctionType type,
                                          const DNA2Parameters& params);
    
    // Phi functions
    static torch::Tensor f5(const torch::Tensor& phi, FunctionType type,
                           const DNA2Parameters& params);
    
    static torch::Tensor f5_derivative(const torch::Tensor& phi, FunctionType type,
                                      const DNA2Parameters& params);
    
    // Utility functions
    static torch::Tensor smooth_step(const torch::Tensor& x, float x0, float x1);
    static torch::Tensor smooth_step_derivative(const torch::Tensor& x, float x0, float x1);

private:
    // Helper functions for piecewise definitions
    static torch::Tensor f1_core(const torch::Tensor& r, FunctionType type,
                                const torch::Tensor& eps,
                                const DNA2Parameters& params);
    
    static torch::Tensor f2_core(const torch::Tensor& r, FunctionType type,
                                const DNA2Parameters& params);
    
    static torch::Tensor f4_core(const torch::Tensor& theta, FunctionType type,
                                const DNA2Parameters& params);
    
    // Parameter access functions
    static float get_f1_parameter(const DNA2Parameters& params, FunctionType type, 
                                 const std::string& param_name);
    static float get_f2_parameter(const DNA2Parameters& params, FunctionType type, 
                                 const std::string& param_name);
    static float get_f4_parameter(const DNA2Parameters& params, FunctionType type, 
                                 const std::string& param_name);
    static float get_f5_parameter(const DNA2Parameters& params, FunctionType type, 
                                 const std::string& param_name);
};

/**
 * @brief Tensor operations utility class
 * 
 * Provides common tensor operations for DNA2 calculations
 */
class TensorOperations {
public:
    // Vector operations
    static torch::Tensor cross_product(const torch::Tensor& a, const torch::Tensor& b);
    static torch::Tensor dot_product(const torch::Tensor& a, const torch::Tensor& b);
    static torch::Tensor normalize(const torch::Tensor& v);
    static torch::Tensor norm(const torch::Tensor& v);
    
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
    
    // Pairwise distance calculations
    static torch::Tensor pairwise_distances(const torch::Tensor& positions);
    static torch::Tensor pairwise_vectors(const torch::Tensor& positions);
    
    // Interaction center calculations
    static torch::Tensor compute_interaction_centers(const torch::Tensor& positions,
                                                    const torch::Tensor& orientations,
                                                    const DNA2Parameters& params);
    
    // Rotation matrix utilities
    static torch::Tensor axis_angle_to_matrix(const torch::Tensor& axis, const torch::Tensor& angle);
    static torch::Tensor matrix_to_axis_angle(const torch::Tensor& matrix);
    static torch::Tensor quaternion_to_matrix(const torch::Tensor& quat);
    static torch::Tensor matrix_to_quaternion(const torch::Tensor& matrix);

private:
    // Helper functions
    static torch::Tensor safe_normalize(const torch::Tensor& v, float eps = 1e-8);
    static torch::Tensor safe_acos(const torch::Tensor& x, float eps = 1e-8);
};

/**
 * @brief Device manager for tensor operations
 * 
 * Manages device selection and tensor device consistency
 */
class DeviceManager {
public:
    static DeviceManager& instance();
    
    void set_device(torch::Device device);
    torch::Device get_device() const { return device_; }
    
    bool is_cuda() const { return device_.is_cuda(); }
    int device_id() const;
    
    // Ensure tensor is on the correct device
    torch::Tensor ensure_device(const torch::Tensor& tensor) const;
    
    // Memory management
    void synchronize_if_cuda() const;
    size_t get_available_memory() const;
    size_t get_total_memory() const;

private:
    DeviceManager() = default;
    torch::Device device_{torch::kCPU};
    
    // Prevent copying
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
};

/**
 * @brief Memory pool for tensor allocation
 * 
 * Provides efficient tensor memory management
 */
class MemoryPool {
private:
    struct TensorInfo {
        torch::Tensor tensor;
        std::vector<int64_t> shape;
        torch::ScalarType dtype;
        torch::Device device;
        bool in_use;
    };
    
    std::vector<TensorInfo> tensor_pool_;
    std::unordered_map<std::string, size_t> tensor_map_;
    size_t max_pool_size_;
    
public:
    explicit MemoryPool(size_t max_pool_size = 1000);
    
    torch::Tensor get_tensor(const std::vector<int64_t>& shape,
                           torch::ScalarType dtype = torch::kFloat32,
                           torch::Device device = torch::kCPU);
    
    void return_tensor(const torch::Tensor& tensor);
    void clear_pool();
    void set_max_pool_size(size_t size) { max_pool_size_ = size; }
    
    // Statistics
    size_t pool_size() const { return tensor_pool_.size(); }
    size_t active_tensors() const;

private:
    std::string tensor_key(const std::vector<int64_t>& shape,
                          torch::ScalarType dtype,
                          torch::Device device) const;
    
    TensorInfo* find_available_tensor(const std::vector<int64_t>& shape,
                                     torch::ScalarType dtype,
                                     torch::Device device);
};

} // namespace dna2