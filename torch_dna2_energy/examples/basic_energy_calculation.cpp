#include "dna2_energy_calculator.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace dna2;

/**
 * @file basic_energy_calculation.cpp
 * @brief Basic example demonstrating single particle pair energy calculation
 * 
 * This example shows how to:
 * - Set up a simple DNA duplex
 * - Calculate energy for a single configuration
 * - Compute forces and verify energy-force consistency
 * - Use different interaction types
 */

void create_simple_dna_duplex(DNAParticle& particles, int strand_length = 10) {
    const int N = particles.num_particles;
    
    std::cout << "Creating DNA duplex with " << N << " nucleotides..." << std::endl;
    
    // Set up positions for two complementary strands
    for (int i = 0; i < N; ++i) {
        float x = (i % strand_length) * 0.7f;  // 0.7 nm spacing along strand
        float y = (i < strand_length) ? 0.0f : 2.0f;  // Two strands separated by 2 nm
        float z = 0.0f;
        
        particles.positions[i][0] = x;
        particles.positions[i][1] = y;
        particles.positions[i][2] = z;
        
        // Set strand IDs
        particles.strand_ids[i] = (i < strand_length) ? 0 : 1;
        
        // Set up complementary base pairing (A-T, G-C)
        int base_idx = i % strand_length;
        if (base_idx % 2 == 0) {
            // A-T pair
            particles.types[i] = (i < strand_length) ? 0 : 1;  // A on strand 0, T on strand 1
            particles.btypes[i] = (i < strand_length) ? 0 : 1;
        } else {
            // G-C pair
            particles.types[i] = (i < strand_length) ? 2 : 3;  // G on strand 0, C on strand 1
            particles.btypes[i] = (i < strand_length) ? 2 : 3;
        }
        
        // Set up backbone bonding (3' to 5' connections within strands)
        if (i < strand_length - 1) {
            // Strand 0 bonding
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        } else if (i >= strand_length && i < N - 1) {
            // Strand 1 bonding
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        }
        
        // Set up simple orientations (identity matrices for simplicity)
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
    }
    
    // Compute interaction centers from positions and orientations
    particles.compute_interaction_centers();
    
    std::cout << "✓ DNA duplex created successfully" << std::endl;
}

void demonstrate_energy_calculation() {
    std::cout << "\n=== Energy Calculation Demo ===" << std::endl;
    
    // Create calculator with default parameters
    auto calculator = create_default_calculator();
    std::cout << "✓ DNA2 energy calculator initialized" << std::endl;
    
    // Create simple DNA duplex
    const int strand_length = 10;
    const int N = 2 * strand_length;  // Two strands
    DNAParticle particles(N, torch::kCPU);
    
    create_simple_dna_duplex(particles, strand_length);
    
    // Validate particle data
    try {
        particles.validate();
        std::cout << "✓ Particle data validation passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Validation failed: " << e.what() << std::endl;
        return;
    }
    
    // Compute total energy
    std::cout << "\nComputing total energy..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    torch::Tensor total_energy = calculator.compute_energy(particles);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Total energy: " << std::fixed << std::setprecision(6) 
              << total_energy.item<float>() << " kT" << std::endl;
    std::cout << "Computation time: " << duration.count() << " microseconds" << std::endl;
    
    // Compute forces
    std::cout << "\nComputing forces..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    torch::Tensor forces = calculator.compute_forces(particles);
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Force computation time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Force tensor shape: " << forces.sizes() << std::endl;
    
    // Display some force components
    std::cout << "Sample force components:" << std::endl;
    for (int i = 0; i < std::min(3, N); ++i) {
        std::cout << "  Particle " << i << ": [" 
                  << std::fixed << std::setprecision(4)
                  << forces[i][0].item<float>() << ", "
                  << forces[i][1].item<float>() << ", "
                  << forces[i][2].item<float>() << "]" << std::endl;
    }
    
    // Compute energy and forces together (more efficient)
    std::cout << "\nComputing energy and forces together..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    auto [energy_joint, forces_joint] = calculator.compute_energy_and_forces(particles);
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Joint computation time: " << duration.count() << " microseconds" << std::endl;
    
    // Verify consistency
    float energy_diff = std::abs(total_energy.item<float>() - energy_joint.item<float>());
    float force_diff = torch::norm(forces - forces_joint).item<float>();
    
    std::cout << "\nConsistency check:" << std::endl;
    std::cout << "Energy difference: " << std::scientific << energy_diff << std::endl;
    std::cout << "Force difference: " << std::scientific << force_diff << std::endl;
    
    if (energy_diff < 1e-6 && force_diff < 1e-6) {
        std::cout << "✓ Consistency check passed" << std::endl;
    } else {
        std::cout << "✗ Consistency check failed" << std::endl;
    }
}

void demonstrate_parameter_effects() {
    std::cout << "\n=== Parameter Effects Demo ===" << std::endl;
    
    // Create test system
    const int N = 20;
    DNAParticle particles(N, torch::kCPU);
    create_simple_dna_duplex(particles, 10);
    
    // Test different temperature values
    std::vector<float> temperatures = {0.1f, 0.5f, 1.0f, 2.0f};
    
    std::cout << "\nTemperature effects on energy:" << std::endl;
    std::cout << std::setw(10) << "Temperature" << std::setw(15) << "Energy (kT)" << std::endl;
    std::cout << std::string(25, '-') << std::endl;
    
    for (float temp : temperatures) {
        DNA2Parameters params;
        params.temperature = temp;
        params.initialize_tensors();
        
        DNA2EnergyCalculator calculator(params);
        torch::Tensor energy = calculator.compute_energy(particles);
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << temp
                  << std::setw(15) << std::setprecision(6) << energy.item<float>() << std::endl;
    }
    
    // Test different salt concentrations
    std::vector<float> salt_concentrations = {0.1f, 0.5f, 1.0f, 2.0f};
    
    std::cout << "\nSalt concentration effects on energy:" << std::endl;
    std::cout << std::setw(15) << "Salt Conc (M)" << std::setw(15) << "Energy (kT)" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    for (float salt : salt_concentrations) {
        DNA2Parameters params;
        params.salt_concentration = salt;
        params.initialize_tensors();
        
        DNA2EnergyCalculator calculator(params);
        torch::Tensor energy = calculator.compute_energy(particles);
        
        std::cout << std::setw(15) << std::fixed << std::setprecision(2) << salt
                  << std::setw(15) << std::setprecision(6) << energy.item<float>() << std::endl;
    }
}

int main() {
    std::cout << "DNA2 Energy Calculator - Basic Energy Calculation Example" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try {
        demonstrate_energy_calculation();
        demonstrate_parameter_effects();
        
        std::cout << "\n✓ All basic energy calculation examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}