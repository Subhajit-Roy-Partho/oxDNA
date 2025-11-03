#pragma once

#include <torch/torch.h>
#include "dna2_data_structures.h"
#include "dna2_interactions.h"
#include "dna2_mesh_interpolation.h"
#include <memory>
#include <vector>

namespace dna2 {

/**
 * @brief Energy calculation pipeline
 * 
 * Orchestrates the complete energy and force calculation process
 */
class EnergyCalculationPipeline {
public:
    explicit EnergyCalculationPipeline(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    /**
     * @brief Compute complete energy and forces for a particle configuration
     * @param particles Particle data
     * @param params DNA2 parameters
     * @param compute_forces Whether to compute forces
     * @param compute_torques Whether to compute torques
     * @return Complete interaction result
     */
    InteractionResult compute_energy(const DNAParticle& particles,
                                   const DNA2Parameters& params,
                                   bool compute_forces = true,
                                   bool compute_torques = true);
    
    /**
     * @brief Compute only energy (faster for energy-only calculations)
     */
    torch::Tensor compute_energy_only(const DNAParticle& particles,
                                    const DNA2Parameters& params);
    
    /**
     * @brief Get interaction manager
     */
    std::shared_ptr<InteractionManager> get_interaction_manager() { return interaction_manager_; }

private:
    std::shared_ptr<InteractionManager> interaction_manager_;
    
    // Distance calculations
    torch::Tensor compute_pairwise_distances(const DNAParticle& particles);
    torch::Tensor compute_interaction_center_distances(const DNAParticle& particles);
    
    // Validation
    void validate_inputs(const DNAParticle& particles, const DNA2Parameters& params);
};

/**
 * @brief Batch processor for multiple configurations
 */
class BatchProcessor {
public:
    explicit BatchProcessor(std::shared_ptr<EnergyCalculationPipeline> pipeline);
    
    /**
     * @brief Process a batch of particle configurations
     * @param batch_particles Vector of particle configurations
     * @param params DNA2 parameters
     * @param compute_forces Whether to compute forces
     * @param compute_torques Whether to compute torques
     * @return Vector of interaction results
     */
    std::vector<InteractionResult> process_batch(
        const std::vector<DNAParticle>& batch_particles,
        const DNA2Parameters& params,
        bool compute_forces = true,
        bool compute_torques = true);
    
    /**
     * @brief Process batch with stacked tensors (more efficient)
     */
    InteractionResult process_batch_stacked(
        const std::vector<DNAParticle>& batch_particles,
        const DNA2Parameters& params,
        bool compute_forces = true,
        bool compute_torques = true);

private:
    std::shared_ptr<EnergyCalculationPipeline> pipeline_;
    
    DNAParticle stack_particles(const std::vector<DNAParticle>& batch);
    std::vector<InteractionResult> unstack_results(const InteractionResult& batch_result,
                                                  int batch_size);
    
    // Batch size optimization
    std::vector<std::vector<int>> optimize_batch_sizes(
        int total_particles,
        int max_batch_size,
        const torch::Device& device);
};

/**
 * @brief Automatic differentiation support
 */
class AutoDiffSupport {
public:
    /**
     * @brief Compute energy with gradient support
     * @param particles Particle data (positions and orientations require grad)
     * @param params DNA2 parameters
     * @return Energy tensor with gradient tracking
     */
    static torch::Tensor compute_energy_with_grad(const DNAParticle& particles,
                                                const DNA2Parameters& params);
    
    /**
     * @brief Compute energy and forces using automatic differentiation
     * @param particles Particle data
     * @param params DNA2 parameters
     * @return Tuple of (energy, forces)
     */
    static std::tuple<torch::Tensor, torch::Tensor> compute_energy_and_forces(
        const DNAParticle& particles,
        const DNA2Parameters& params);
    
    /**
     * @brief Compute energy, forces, and torques using automatic differentiation
     * @param particles Particle data
     * @param params DNA2 parameters
     * @return Tuple of (energy, forces, torques)
     */
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
    compute_energy_forces_and_torques(
        const DNAParticle& particles,
        const DNA2Parameters& params);

private:
    static torch::Tensor energy_function(const torch::Tensor& positions,
                                       const torch::Tensor& orientations,
                                       const DNA2Parameters& params);
    
    static torch::Tensor compute_torques_from_orientation_grad(
        const torch::Tensor& orientation_grad,
        const torch::Tensor& orientations);
};

/**
 * @brief Performance profiler for energy calculations
 */
class PerformanceProfiler {
private:
    struct TimingInfo {
        std::chrono::high_resolution_clock::time_point start_time;
        double accumulated_time;
        int call_count;
    };
    
    std::unordered_map<std::string, TimingInfo> timings_;
    bool enabled_;

public:
    explicit PerformanceProfiler(bool enabled = false) : enabled_(enabled) {}
    
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }
    
    void start_timing(const std::string& operation);
    void end_timing(const std::string& operation);
    
    void print_report() const;
    void reset();
    
    // Get timing statistics
    double get_total_time(const std::string& operation) const;
    double get_average_time(const std::string& operation) const;
    int get_call_count(const std::string& operation) const;
};

/**
 * @brief Input validator for particle and parameter data
 */
class InputValidator {
public:
    /**
     * @brief Validate particle data structure
     */
    static void validate_particles(const DNAParticle& particles);
    
    /**
     * @brief Validate parameter structure
     */
    static void validate_parameters(const DNA2Parameters& params);
    
    /**
     * @brief Validate device compatibility
     */
    static void validate_device_compatibility(const torch::Device& device);
    
    /**
     * @brief Validate tensor shapes
     */
    static void validate_tensor_shapes(const torch::Tensor& tensor,
                                     const std::vector<int64_t>& expected_shape,
                                     const std::string& tensor_name);
    
    /**
     * @brief Validate tensor device
     */
    static void validate_tensor_device(const torch::Tensor& tensor,
                                     const torch::Device& expected_device,
                                     const std::string& tensor_name);

private:
    static void check_tensor_shapes(const torch::Tensor& tensor,
                                  const std::vector<int64_t>& expected_shape);
    static void check_tensor_device(const torch::Tensor& tensor,
                                  const torch::Device& expected_device);
};

/**
 * @brief Utility functions for energy calculations
 */
class EnergyUtils {
public:
    /**
     * @brief Convert between different coordinate representations
     */
    static DNAParticle from_oxdna_format(const std::vector<std::tuple<torch::Tensor, torch::Tensor, int, int>>& oxdna_particles,
                                       torch::Device device = torch::kCPU);
    
    static std::vector<std::tuple<torch::Tensor, torch::Tensor, int, int>> 
    to_oxdna_format(const DNAParticle& torch_particles);
    
    /**
     * @brief I/O utilities
     */
    static void save_configuration(const DNAParticle& particles, const std::string& filename);
    static DNAParticle load_configuration(const std::string& filename, torch::Device device = torch::kCPU);
    
    /**
     * @brief Energy decomposition utilities
     */
    static std::unordered_map<std::string, torch::Tensor> 
    decompose_energy(const InteractionResult& result);
    
    static torch::Tensor compute_energy_breakdown(const DNAParticle& particles,
                                                const DNA2Parameters& params);
    
    /**
     * @brief Analysis utilities
     */
    static torch::Tensor compute_energy_per_particle(const InteractionResult& result);
    static torch::Tensor compute_force_magnitudes(const torch::Tensor& forces);
    static torch::Tensor compute_torque_magnitudes(const torch::Tensor& torques);
    
    /**
     * @brief Configuration analysis
     */
    static torch::Tensor compute_radius_of_gyration(const DNAParticle& particles);
    static torch::Tensor compute_end_to_end_distance(const DNAParticle& particles);
    static torch::Tensor compute_base_pairing_map(const DNAParticle& particles);

private:
    // Helper functions for coordinate conversions
    static torch::Tensor quaternion_to_rotation_matrix(const torch::Tensor& quat);
    static torch::Tensor rotation_matrix_to_quaternion(const torch::Tensor& matrix);
};

} // namespace dna2