#include "dna2_energy_calculator.h"
#include <iostream>
#include <chrono>

using namespace dna2;

int main() {
    std::cout << "DNA2 Energy Calculator Example" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        // Initialize parameters
        DNA2Parameters params;
        params.temperature = 0.1f;
        params.salt_concentration = 0.5f;
        
        // Create calculator (CPU version)
        auto calculator = create_default_calculator();
        std::cout << "Created DNA2 energy calculator on CPU" << std::endl;
        
        // Try to create GPU version if available
        try {
            auto gpu_calculator = create_gpu_calculator();
            std::cout << "Created DNA2 energy calculator on GPU" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "GPU not available: " << e.what() << std::endl;
        }
        
        // Create a simple DNA duplex (2 strands, 10 nucleotides each)
        const int N = 20;  // Total number of nucleotides
        const int strand_length = 10;
        
        DNAParticle particles(N, torch::kCPU);
        
        // Set up positions (simple linear arrangement)
        for (int i = 0; i < N; ++i) {
            float x = (i % strand_length) * 0.7f;  // 0.7 nm spacing
            float y = (i < strand_length) ? 0.0f : 2.0f;  // Two strands separated by 2 nm
            float z = 0.0f;
            
            particles.positions[i][0] = x;
            particles.positions[i][1] = y;
            particles.positions[i][2] = z;
            
            // Set strand IDs
            particles.strand_ids[i] = (i < strand_length) ? 0 : 1;
            
            // Set base types (simplified: alternating A-T and G-C)
            if (i % 2 == 0) {
                particles.types[i] = 0;  // A
                particles.btypes[i] = 0;  // A
            } else {
                particles.types[i] = 1;  // T
                particles.btypes[i] = 1;  // T
            }
            
            // Set up bonding (3' to 5' connections within strands)
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
            
            // Set up simple orientations (identity matrices for simplicity)
            particles.orientations[i] = torch::eye(3, torch::kFloat32);
        }
        
        // Compute interaction centers
        particles.compute_interaction_centers();
        
        std::cout << "Created DNA duplex with " << N << " nucleotides" << std::endl;
        
        // Validate particle data
        particles.validate();
        std::cout << "Particle data validation passed" << std::endl;
        
        // Compute energy
        std::cout << "\nComputing energy..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        torch::Tensor energy = calculator.compute_energy(particles);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Total energy: " << energy.item<float>() << std::endl;
        std::cout << "Computation time: " << duration.count() << " microseconds" << std::endl;
        
        // Compute forces
        std::cout << "\nComputing forces..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        torch::Tensor forces = calculator.compute_forces(particles);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Force computation time: " << duration.count() << " microseconds" << std::endl;
        
        // Compute energy and forces together
        std::cout << "\nComputing energy and forces together..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        auto [energy2, forces2] = calculator.compute_energy_and_forces(particles);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Energy (joint): " << energy2.item<float>() << std::endl;
        std::cout << "Joint computation time: " << duration.count() << " microseconds" << std::endl;
        
        // Verify consistency
        float energy_diff = std::abs(energy.item<float>() - energy2.item<float>());
        float force_diff = torch::norm(forces - forces2).item<float>();
        
        std::cout << "Energy difference: " << energy_diff << std::endl;
        std::cout << "Force difference: " << force_diff << std::endl;
        
        if (energy_diff < 1e-6 && force_diff < 1e-6) {
            std::cout << "✓ Consistency check passed" << std::endl;
        } else {
            std::cout << "✗ Consistency check failed" << std::endl;
        }
        
        // Test batch processing
        std::cout << "\nTesting batch processing..." << std::endl;
        std::vector<DNAParticle> batch;
        for (int i = 0; i < 3; ++i) {
            batch.push_back(particles);  // Same configuration 3 times
        }
        
        start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor batch_energy = calculator.compute_energy_batch(batch);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Batch energy shape: " << batch_energy.sizes() << std::endl;
        std::cout << "Batch computation time: " << duration.count() << " microseconds" << std::endl;
        
        // Test automatic differentiation
        std::cout << "\nTesting automatic differentiation..." << std::endl;
        
        // Enable gradients for positions
        auto positions_grad = particles.positions.clone().detach().set_requires_grad(true);
        auto orientations_grad = particles.orientations.clone().detach().set_requires_grad(true);
        
        start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor energy_grad = calculator.compute_energy_autograd(positions_grad, orientations_grad);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Energy with grad: " << energy_grad.item<float>() << std::endl;
        std::cout << "Autograd computation time: " << duration.count() << " microseconds" << std::endl;
        
        // Compute gradients
        energy_grad.backward();
        torch::Tensor position_gradients = positions_grad.grad();
        
        std::cout << "Position gradients shape: " << position_gradients.sizes() << std::endl;
        std::cout << "Max gradient magnitude: " << torch::max(torch::abs(position_gradients)).item<float>() << std::endl;
        
        std::cout << "\n✓ All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}