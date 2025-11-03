#pragma once

#include <torch/torch.h>
#include "dna2_data_structures.h"
#include <unordered_map>

namespace dna2 {

/**
 * @brief Mesh data structure for angular potential interpolation
 */
struct MeshData {
    torch::Tensor x_values;      // [N_points] - cosine values
    torch::Tensor y_values;      // [N_points] - function values
    torch::Tensor y_derivatives; // [N_points] - derivative values
    torch::Tensor tangents;      // [N_points] - Hermite tangents
    int mesh_type;
    int n_points;
    
    MeshData() = default;
    MeshData(int type, int n_pts, torch::Device device);
    
    void to(torch::Device device);
};

/**
 * @brief Mesh interpolator for angular potentials
 * 
 * Provides cubic Hermite interpolation for angular potential functions
 * used in DNA2 energy calculations.
 */
class MeshInterpolator {
private:
    std::unordered_map<int, MeshData> meshes_;
    bool initialized_;
    torch::Device device_;
    
public:
    explicit MeshInterpolator(torch::Device device = torch::kCPU);
    
    /**
     * @brief Initialize all mesh types
     * @param params DNA2 parameters
     */
    void initialize_meshes(const DNA2Parameters& params);
    
    /**
     * @brief Interpolate function value at given cosine values
     * @param mesh_type Type of mesh to use
     * @param cos_values Cosine values to interpolate at
     * @return Interpolated function values
     */
    torch::Tensor interpolate(MeshType mesh_type, const torch::Tensor& cos_values);
    
    /**
     * @brief Interpolate derivative at given cosine values
     * @param mesh_type Type of mesh to use
     * @param cos_values Cosine values to interpolate at
     * @return Interpolated derivative values
     */
    torch::Tensor interpolate_derivative(MeshType mesh_type, const torch::Tensor& cos_values);
    
    /**
     * @brief Check if meshes are initialized
     */
    bool is_initialized() const { return initialized_; }
    
    /**
     * @brief Get mesh data for a specific type
     */
    const MeshData& get_mesh_data(MeshType mesh_type) const;
    
    /**
     * @brief Move all meshes to specified device
     */
    void to(torch::Device device);

private:
    // Initialize individual mesh types
    void initialize_hydr_meshes(const DNA2Parameters& params);
    void initialize_stck_meshes(const DNA2Parameters& params);
    void initialize_crst_meshes(const DNA2Parameters& params);
    void initialize_cxst_meshes(const DNA2Parameters& params);
    
    void initialize_single_mesh(MeshType mesh_type, const DNA2Parameters& params,
                               int n_points, float t0, float tc, float ts);
    void initialize_cxst_theta1_mesh(const DNA2Parameters& params);
    
    // Hermite interpolation
    torch::Tensor cubic_hermite_interpolate(const torch::Tensor& x,
                                          const torch::Tensor& y,
                                          const torch::Tensor& dy,
                                          const torch::Tensor& query_points);
    
    torch::Tensor cubic_hermite_interpolate_derivative(const torch::Tensor& x,
                                                     const torch::Tensor& y,
                                                     const torch::Tensor& dy,
                                                     const torch::Tensor& query_points);
    
    // Tangent computation
    torch::Tensor compute_hermite_tangents(const torch::Tensor& x,
                                          const torch::Tensor& y,
                                          const torch::Tensor& dy);
    
    // Function computation for mesh generation
    torch::Tensor compute_f4_function(const torch::Tensor& theta, MeshType mesh_type,
                                     const DNA2Parameters& params);
    torch::Tensor compute_f4_derivative(const torch::Tensor& theta, MeshType mesh_type,
                                       const DNA2Parameters& params);
    
    // Mesh parameters
    static constexpr int HYDR_T1_MESH_POINTS = 200;
    static constexpr int HYDR_T2_MESH_POINTS = 200;
    static constexpr int HYDR_T3_MESH_POINTS = 200;
    static constexpr int HYDR_T4_MESH_POINTS = 200;
    static constexpr int HYDR_T7_MESH_POINTS = 200;
    static constexpr int HYDR_T8_MESH_POINTS = 200;
    
    static constexpr int STCK_T1_MESH_POINTS = 200;
    static constexpr int STCK_T2_MESH_POINTS = 200;
    static constexpr int STCK_T3_MESH_POINTS = 200;
    static constexpr int STCK_T4_MESH_POINTS = 200;
    
    static constexpr int CRST_T1_MESH_POINTS = 200;
    static constexpr int CRST_T2_MESH_POINTS = 200;
    
    static constexpr int CXST_T1_MESH_POINTS = 200;
    
    // Mesh angle parameters (in radians)
    static constexpr float HYDR_THETA1_T0 = 2.35619449f;  // 135 degrees
    static constexpr float HYDR_THETA1_TC = 0.785398163f; // 45 degrees
    static constexpr float HYDR_THETA1_TS = 0.174532925f; // 10 degrees
    
    static constexpr float HYDR_THETA2_T0 = 2.35619449f;  // 135 degrees
    static constexpr float HYDR_THETA2_TC = 0.785398163f; // 45 degrees
    static constexpr float HYDR_THETA2_TS = 0.174532925f; // 10 degrees
    
    static constexpr float HYDR_THETA3_T0 = 0.785398163f;  // 45 degrees
    static constexpr float HYDR_THETA3_TC = 0.785398163f;  // 45 degrees
    static constexpr float HYDR_THETA3_TS = 0.174532925f;  // 10 degrees
    
    static constexpr float HYDR_THETA4_T0 = 0.0f;           // 0 degrees
    static constexpr float HYDR_THETA4_TC = 0.785398163f;   // 45 degrees
    static constexpr float HYDR_THETA4_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float HYDR_THETA7_T0 = 0.785398163f;   // 45 degrees
    static constexpr float HYDR_THETA7_TC = 0.785398163f;   // 45 degrees
    static constexpr float HYDR_THETA7_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float HYDR_THETA8_T0 = 0.0f;           // 0 degrees
    static constexpr float HYDR_THETA8_TC = 0.785398163f;   // 45 degrees
    static constexpr float HYDR_THETA8_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float STCK_THETA1_T0 = 0.959931089f;   // 55 degrees
    static constexpr float STCK_THETA1_TC = 0.523598776f;   // 30 degrees
    static constexpr float STCK_THETA1_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float STCK_THETA2_T0 = 0.959931089f;   // 55 degrees
    static constexpr float STCK_THETA2_TC = 0.523598776f;   // 30 degrees
    static constexpr float STCK_THETA2_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float STCK_THETA3_T0 = 0.0f;           // 0 degrees
    static constexpr float STCK_THETA3_TC = 0.523598776f;   // 30 degrees
    static constexpr float STCK_THETA3_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float STCK_THETA4_T0 = 0.0f;           // 0 degrees
    static constexpr float STCK_THETA4_TC = 0.523598776f;   // 30 degrees
    static constexpr float STCK_THETA4_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float CRST_THETA1_T0 = 0.523598776f;   // 30 degrees
    static constexpr float CRST_THETA1_TC = 0.523598776f;   // 30 degrees
    static constexpr float CRST_THETA1_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float CRST_THETA2_T0 = 0.0f;           // 0 degrees
    static constexpr float CRST_THETA2_TC = 0.523598776f;   // 30 degrees
    static constexpr float CRST_THETA2_TS = 0.174532925f;   // 10 degrees
    
    static constexpr float CXST_THETA1_T0 = 0.0f;           // 0 degrees
    static constexpr float CXST_THETA1_TC = 0.523598776f;   // 30 degrees
    static constexpr float CXST_THETA1_TS = 0.174532925f;   // 10 degrees
};

/**
 * @brief Mesh factory for creating pre-computed meshes
 */
class MeshFactory {
public:
    /**
     * @brief Create a mesh interpolator with default parameters
     */
    static std::unique_ptr<MeshInterpolator> create_default_interpolator(
        torch::Device device = torch::kCPU);
    
    /**
     * @brief Create a mesh interpolator with custom parameters
     */
    static std::unique_ptr<MeshInterpolator> create_interpolator(
        const DNA2Parameters& params,
        torch::Device device = torch::kCPU);
    
    /**
     * @brief Load meshes from file (if available)
     */
    static std::unique_ptr<MeshInterpolator> load_from_file(
        const std::string& filename,
        torch::Device device = torch::kCPU);
    
    /**
     * @brief Save meshes to file
     */
    static void save_to_file(const MeshInterpolator& interpolator,
                           const std::string& filename);

private:
    // Helper functions for file I/O
    static void serialize_mesh(const MeshData& mesh, std::ostream& os);
    static MeshData deserialize_mesh(std::istream& os, torch::Device device);
};

} // namespace dna2