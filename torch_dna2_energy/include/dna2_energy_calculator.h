#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <exception>

namespace dna2 {

// Forward declarations
class DNA2Parameters;
class DNAParticle;
class InteractionResult;
class EnergyCalculationPipeline;
class InteractionManager;
class DeviceManager;
class MemoryPool;

/**
 * @brief Main interface class for DNA2 energy calculator
 * 
 * This class provides the primary interface for computing DNA2 energies and forces
 * using PyTorch C++ tensors with GPU acceleration support.
 */
class DNA2EnergyCalculator {
public:
    /**
     * @brief Constructor
     * @param params DNA2 model parameters
     * @param device PyTorch device (CPU or CUDA)
     */
    explicit DNA2EnergyCalculator(const DNA2Parameters& params = DNA2Parameters{},
                                 torch::Device device = torch::kCPU);
    
    /**
     * @brief Destructor
     */
    ~DNA2EnergyCalculator();
    
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
    torch::Device device() const;

private:
    DNA2Parameters params_;
    std::unique_ptr<DeviceManager> device_manager_;
    std::unique_ptr<EnergyCalculationPipeline> pipeline_;
    std::unique_ptr<MemoryPool> memory_pool_;
    
    void initialize_components();
};

/**
 * @brief Custom exception for DNA2 energy calculator errors
 */
class DNA2EnergyException : public std::exception {
private:
    std::string message_;
public:
    explicit DNA2EnergyException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
};

// Factory functions
DNA2EnergyCalculator create_default_calculator();
DNA2EnergyCalculator create_gpu_calculator(int gpu_id = 0);

} // namespace dna2