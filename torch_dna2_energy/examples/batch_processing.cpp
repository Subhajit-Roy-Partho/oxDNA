#include "dna2_energy_calculator.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace dna2;

/**
 * @file batch_processing.cpp
 * @brief Example demonstrating batch processing of multiple DNA configurations
 * 
 * This example shows how to:
 * - Process multiple DNA configurations efficiently
 * - Compare batch vs individual processing performance
 * - Handle different system sizes in batches
 * - Use batch processing for parameter sweeps
 */

void create_dna_configuration(DNAParticle& particles, int strand_length, 
                             float separation = 2.0f, float rotation = 0.0f) {
    const int N = particles.num_particles;
    
    // Set up positions for two strands with configurable separation and rotation
    for (int i = 0; i < N; ++i) {
        float x = (i % strand_length) * 0.7f;
        float y = (i < strand_length) ? 0.0f : separation;
        float z = 0.0f;
        
        // Apply rotation to second strand
        if (i >= strand_length) {
            float cos_rot = std::cos(rotation);
            float sin_rot = std::sin(rotation);
            float x_rot = x * cos_rot - z * sin_rot;
            float z_rot = x * sin_rot + z * cos_rot;
            x = x_rot;
            z = z_rot;
        }
        
        particles.positions[i][0] = x;
        particles.positions[i][1] = y;
        particles.positions[i][2] = z;
        
        // Set strand IDs
        particles.strand_ids[i] = (i < strand_length) ? 0 : 1;
        
        // Set up complementary base pairing
        int base_idx = i % strand_length;
        if (base_idx % 2 == 0) {
            particles.types[i] = (i < strand_length) ? 0 : 1;  // A-T
            particles.btypes[i] = (i < strand_length) ? 0 : 1;
        } else {
            particles.types[i] = (i < strand_length) ? 2 : 3;  // G-C
            particles.btypes[i] = (i < strand_length) ? 2 : 3;
        }
        
        // Set up backbone bonding
        if (i < strand_length - 1) {
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        } else if (i >= strand_length && i < N - 1) {
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        }
        
        // Set up orientations
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
    }
    
    particles.compute_interaction_centers();
}

void demonstrate_batch_vs_individual() {
    std::cout << "\n=== Batch vs Individual Processing Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 10;
    const int N = 2 * strand_length;
    const int batch_size = 8;
    
    // Create batch of configurations with different strand separations
    std::vector<DNAParticle> batch;
    std::vector<float> separations = {1.5f, 1.8f, 2.0f, 2.2f, 2.5f, 2.8f, 3.0f, 3.5f};
    
    std::cout << "Creating batch of " << batch_size << " configurations..." << std::endl;
    for (int i = 0; i < batch_size; ++i) {
        DNAParticle particles(N, torch::kCPU);
        create_dna_configuration(particles, strand_length, separations[i]);
        batch.push_back(particles);
    }
    
    // Process individually
    std::cout << "\nProcessing configurations individually..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<float> individual_energies;
    for (const auto& particles : batch) {
        torch::Tensor energy = calculator.compute_energy(particles);
        individual_energies.push_back(energy.item<float>());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Process as batch
    std::cout << "Processing configurations as batch..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    torch::Tensor batch_energies = calculator.compute_energy_batch(batch);
    
    end_time = std::chrono::high_resolution_clock::now();
    auto batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Compare results
    std::cout << "\nResults comparison:" << std::endl;
    std::cout << std::setw(10) << "Separation" << std::setw(15) << "Individual" 
              << std::setw(15) << "Batch" << std::setw(15) << "Difference" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    float max_diff = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        float diff = std::abs(individual_energies[i] - batch_energies[i].item<float>());
        max_diff = std::max(max_diff, diff);
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << separations[i]
                  << std::setw(15) << std::setprecision(6) << individual_energies[i]
                  << std::setw(15) << std::setprecision(6) << batch_energies[i].item<float>()
                  << std::setw(15) << std::scientific << diff << std::endl;
    }
    
    std::cout << "\nPerformance comparison:" << std::endl;
    std::cout << "Individual processing time: " << individual_time.count() << " microseconds" << std::endl;
    std::cout << "Batch processing time: " << batch_time.count() << " microseconds" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
              << static_cast<float>(individual_time.count()) / batch_time.count() << "x" << std::endl;
    std::cout << "Maximum difference: " << std::scientific << max_diff << std::endl;
    
    if (max_diff < 1e-6) {
        std::cout << "✓ Batch and individual results are consistent" << std::endl;
    } else {
        std::cout << "✗ Significant differences detected" << std::endl;
    }
}

void demonstrate_parameter_sweep() {
    std::cout << "\n=== Parameter Sweep Demo ===" << std::endl;
    
    const int strand_length = 8;
    const int N = 2 * strand_length;
    
    // Create base configuration
    DNAParticle base_particles(N, torch::kCPU);
    create_dna_configuration(base_particles, strand_length, 2.0f);
    
    // Temperature sweep
    std::vector<float> temperatures = {0.1f, 0.2f, 0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<DNAParticle> temp_batch;
    
    for (size_t i = 0; i < temperatures.size(); ++i) {
        temp_batch.push_back(base_particles);  // Same geometry, different parameters
    }
    
    std::cout << "Performing temperature sweep with " << temperatures.size() << " values..." << std::endl;
    
    // Process with different temperature parameters
    std::vector<float> temp_energies;
    for (size_t i = 0; i < temperatures.size(); ++i) {
        DNA2Parameters params;
        params.temperature = temperatures[i];
        params.initialize_tensors();
        
        DNA2EnergyCalculator calculator(params);
        torch::Tensor energy = calculator.compute_energy(temp_batch[i]);
        temp_energies.push_back(energy.item<float>());
    }
    
    std::cout << "\nTemperature sweep results:" << std::endl;
    std::cout << std::setw(12) << "Temperature" << std::setw(15) << "Energy (kT)" << std::endl;
    std::cout << std::string(27, '-') << std::endl;
    
    for (size_t i = 0; i < temperatures.size(); ++i) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(2) << temperatures[i]
                  << std::setw(15) << std::setprecision(6) << temp_energies[i] << std::endl;
    }
    
    // Salt concentration sweep
    std::vector<float> salt_concentrations = {0.05f, 0.1f, 0.2f, 0.5f, 1.0f, 2.0f};
    std::vector<DNAParticle> salt_batch;
    
    for (size_t i = 0; i < salt_concentrations.size(); ++i) {
        salt_batch.push_back(base_particles);
    }
    
    std::cout << "\nPerforming salt concentration sweep with " << salt_concentrations.size() << " values..." << std::endl;
    
    std::vector<float> salt_energies;
    for (size_t i = 0; i < salt_concentrations.size(); ++i) {
        DNA2Parameters params;
        params.salt_concentration = salt_concentrations[i];
        params.initialize_tensors();
        
        DNA2EnergyCalculator calculator(params);
        torch::Tensor energy = calculator.compute_energy(salt_batch[i]);
        salt_energies.push_back(energy.item<float>());
    }
    
    std::cout << "\nSalt concentration sweep results:" << std::endl;
    std::cout << std::setw(15) << "Salt (M)" << std::setw(15) << "Energy (kT)" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    for (size_t i = 0; i < salt_concentrations.size(); ++i) {
        std::cout << std::setw(15) << std::fixed << std::setprecision(3) << salt_concentrations[i]
                  << std::setw(15) << std::setprecision(6) << salt_energies[i] << std::endl;
    }
}

void demonstrate_different_system_sizes() {
    std::cout << "\n=== Different System Sizes Demo ===" << std::endl;
    
    // Create systems of different sizes
    std::vector<int> strand_lengths = {5, 10, 15, 20};
    std::vector<DNAParticle> size_batch;
    
    for (int length : strand_lengths) {
        int N = 2 * length;
        DNAParticle particles(N, torch::kCPU);
        create_dna_configuration(particles, length, 2.0f);
        size_batch.push_back(particles);
    }
    
    std::cout << "Processing systems of different sizes..." << std::endl;
    std::cout << std::setw(15) << "System Size" << std::setw(15) << "Energy (kT)" 
              << std::setw(20) << "Time (μs)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    auto calculator = create_default_calculator();
    
    for (size_t i = 0; i < size_batch.size(); ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        torch::Tensor energy = calculator.compute_energy(size_batch[i]);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << std::setw(15) << size_batch[i].num_particles
                  << std::setw(15) << std::fixed << std::setprecision(6) << energy.item<float>()
                  << std::setw(20) << duration.count() << std::endl;
    }
    
    // Batch process all systems (note: this would require same size in real implementation)
    std::cout << "\nNote: Batch processing requires systems of the same size." << std::endl;
    std::cout << "For different sizes, process individually or use padding." << std::endl;
}

int main() {
    std::cout << "DNA2 Energy Calculator - Batch Processing Example" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        demonstrate_batch_vs_individual();
        demonstrate_parameter_sweep();
        demonstrate_different_system_sizes();
        
        std::cout << "\n✓ All batch processing examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}