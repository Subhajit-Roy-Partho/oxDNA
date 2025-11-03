#include "dna2_energy_calculator.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace dna2;

/**
 * @file gpu_acceleration.cpp
 * @brief Example demonstrating GPU acceleration and performance benefits
 * 
 * This example shows how to:
 * - Use GPU acceleration for energy calculations
 * - Compare CPU vs GPU performance
 * - Handle memory management between devices
 * - Optimize batch processing for GPU
 */

void create_large_dna_system(DNAParticle& particles, int num_strands, int strand_length) {
    const int N = particles.num_particles;
    
    std::cout << "Creating large DNA system: " << num_strands << " strands × " 
              << strand_length << " nucleotides each..." << std::endl;
    
    // Create multiple strands arranged in a grid
    int grid_size = static_cast<int>(std::sqrt(num_strands)) + 1;
    float spacing = 3.0f;  // Spacing between strands
    
    for (int i = 0; i < N; ++i) {
        int strand_id = i / strand_length;
        int pos_in_strand = i % strand_length;
        
        // Arrange strands in a grid
        int grid_x = strand_id % grid_size;
        int grid_y = strand_id / grid_size;
        
        float x = pos_in_strand * 0.7f + grid_x * spacing * strand_length * 0.7f;
        float y = grid_y * spacing;
        float z = 0.0f;
        
        particles.positions[i][0] = x;
        particles.positions[i][1] = y;
        particles.positions[i][2] = z;
        
        // Set strand IDs
        particles.strand_ids[i] = strand_id;
        
        // Random base types for diversity
        particles.types[i] = pos_in_strand % 4;
        particles.btypes[i] = pos_in_strand % 4;
        
        // Set up backbone bonding within strands
        if (pos_in_strand < strand_length - 1) {
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        }
        
        // Set up orientations (slightly varied for realism)
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
        if (pos_in_strand % 5 == 0) {
            // Add slight rotation every 5 nucleotides
            float angle = 0.1f;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            particles.orientations[i][0][0] = cos_a;
            particles.orientations[i][0][2] = sin_a;
            particles.orientations[i][2][0] = -sin_a;
            particles.orientations[i][2][2] = cos_a;
        }
    }
    
    particles.compute_interaction_centers();
    std::cout << "✓ Large DNA system created successfully" << std::endl;
}

void compare_cpu_gpu_performance() {
    std::cout << "\n=== CPU vs GPU Performance Comparison ===" << std::endl;
    
    // Test different system sizes
    std::vector<std::pair<int, int>> test_sizes = {
        {2, 20},    // 2 strands × 20 nucleotides = 40 particles
        {4, 25},    // 4 strands × 25 nucleotides = 100 particles
        {8, 25},    // 8 strands × 25 nucleotides = 200 particles
        {16, 25},   // 16 strands × 25 nucleotides = 400 particles
        {32, 25}    // 32 strands × 25 nucleotides = 800 particles
    };
    
    std::cout << std::setw(12) << "Particles" << std::setw(15) << "CPU (μs)" 
              << std::setw(15) << "GPU (μs)" << std::setw(12) << "Speedup" 
              << std::setw(15) << "CPU Energy" << std::setw(15) << "GPU Energy" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    for (const auto& size : test_sizes) {
        int num_strands = size.first;
        int strand_length = size.second;
        int N = num_strands * strand_length;
        
        // Create test system
        DNAParticle particles(N, torch::kCPU);
        create_large_dna_system(particles, num_strands, strand_length);
        
        // CPU performance test
        DNA2EnergyCalculator cpu_calculator;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor cpu_energy = cpu_calculator.compute_energy(particles);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // GPU performance test (if available)
        float gpu_time_us = -1.0f;
        float gpu_energy_val = -1.0f;
        
        try {
            auto gpu_calculator = create_gpu_calculator();
            
            // Move data to GPU
            DNAParticle gpu_particles = particles;
            gpu_particles.to(torch::kCUDA);
            
            start_time = std::chrono::high_resolution_clock::now();
            torch::Tensor gpu_energy = gpu_calculator.compute_energy(gpu_particles);
            end_time = std::chrono::high_resolution_clock::now();
            auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            gpu_time_us = static_cast<float>(gpu_time.count());
            gpu_energy_val = gpu_energy.cpu().item<float>();
            
        } catch (const std::exception& e) {
            std::cout << "GPU not available: " << e.what() << std::endl;
        }
        
        // Calculate speedup
        float speedup = (gpu_time_us > 0) ? static_cast<float>(cpu_time.count()) / gpu_time_us : 0.0f;
        
        // Display results
        std::cout << std::setw(12) << N
                  << std::setw(15) << cpu_time.count();
        
        if (gpu_time_us > 0) {
            std::cout << std::setw(15) << static_cast<int>(gpu_time_us)
                      << std::setw(12) << std::fixed << std::setprecision(2) << speedup;
        } else {
            std::cout << std::setw(15) << "N/A"
                      << std::setw(12) << "N/A";
        }
        
        std::cout << std::setw(15) << std::fixed << std::setprecision(6) << cpu_energy.item<float>();
        
        if (gpu_energy_val > 0) {
            float energy_diff = std::abs(cpu_energy.item<float>() - gpu_energy_val);
            std::cout << std::setw(15) << std::setprecision(6) << gpu_energy_val;
            if (energy_diff > 1e-5) {
                std::cout << " (diff: " << std::scientific << energy_diff << ")";
            }
        } else {
            std::cout << std::setw(15) << "N/A";
        }
        
        std::cout << std::endl;
    }
}

void demonstrate_gpu_memory_management() {
    std::cout << "\n=== GPU Memory Management Demo ===" << std::endl;
    
    try {
        // Create GPU calculator
        auto gpu_calculator = create_gpu_calculator();
        std::cout << "✓ GPU calculator created successfully" << std::endl;
        
        // Create test system on CPU
        const int N = 200;
        DNAParticle cpu_particles(N, torch::kCPU);
        create_large_dna_system(cpu_particles, 8, 25);
        
        std::cout << "Original device: " << cpu_particles.positions.device() << std::endl;
        
        // Move to GPU
        DNAParticle gpu_particles = cpu_particles;
        gpu_particles.to(torch::kCUDA);
        
        std::cout << "After transfer: " << gpu_particles.positions.device() << std::endl;
        
        // Compute energy on GPU
        torch::Tensor gpu_energy = gpu_calculator.compute_energy(gpu_particles);
        std::cout << "GPU energy: " << gpu_energy.item<float>() << " kT" << std::endl;
        
        // Move result back to CPU
        torch::Tensor cpu_energy = gpu_energy.cpu();
        std::cout << "Energy on CPU: " << cpu_energy.item<float>() << " kT" << std::endl;
        
        // Test batch processing on GPU
        std::vector<DNAParticle> batch;
        for (int i = 0; i < 4; ++i) {
            DNAParticle batch_particles(N, torch::kCPU);
            create_large_dna_system(batch_particles, 8, 25);
            
            // Add slight variation to each batch
            batch_particles.positions += torch::rand_like(batch_particles.positions) * 0.1f;
            batch_particles.compute_interaction_centers();
            
            // Move to GPU
            batch_particles.to(torch::kCUDA);
            batch.push_back(batch_particles);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor batch_energy = gpu_calculator.compute_energy_batch(batch);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "GPU batch processing time: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Batch energies: " << batch_energy.cpu() << std::endl;
        
        std::cout << "✓ GPU memory management demo completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "GPU demo failed: " << e.what() << std::endl;
        std::cout << "This is expected if CUDA is not available" << std::endl;
    }
}

void demonstrate_optimization_techniques() {
    std::cout << "\n=== GPU Optimization Techniques Demo ===" << std::endl;
    
    const int N = 400;
    DNAParticle particles(N, torch::kCPU);
    create_large_dna_system(particles, 16, 25);
    
    // Test different batch sizes for optimal GPU utilization
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32};
    
    std::cout << "\nBatch size optimization:" << std::endl;
    std::cout << std::setw(12) << "Batch Size" << std::setw(15) << "Time (μs)" 
              << std::setw(15) << "Time per item" << std::endl;
    std::cout << std::string(42, '-') << std::endl;
    
    try {
        auto gpu_calculator = create_gpu_calculator();
        
        for (int batch_size : batch_sizes) {
            // Create batch
            std::vector<DNAParticle> batch;
            for (int i = 0; i < batch_size; ++i) {
                DNAParticle batch_particles = particles;
                batch_particles.positions += torch::rand_like(batch_particles.positions) * 0.05f;
                batch_particles.compute_interaction_centers();
                batch_particles.to(torch::kCUDA);
                batch.push_back(batch_particles);
            }
            
            // Time batch processing
            auto start_time = std::chrono::high_resolution_clock::now();
            torch::Tensor batch_energy = gpu_calculator.compute_energy_batch(batch);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            float time_per_item = static_cast<float>(duration.count()) / batch_size;
            
            std::cout << std::setw(12) << batch_size
                      << std::setw(15) << duration.count()
                      << std::setw(15) << std::fixed << std::setprecision(2) << time_per_item << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "GPU optimization demo skipped: " << e.what() << std::endl;
    }
    
    // Demonstrate memory pooling benefits
    std::cout << "\nMemory pooling benefits:" << std::endl;
    std::cout << "The calculator uses memory pooling to reduce allocation overhead." << std::endl;
    std::cout << "Repeated calculations with the same system size are faster due to" << std::endl;
    std::cout << "reused tensor memory allocations." << std::endl;
}

int main() {
    std::cout << "DNA2 Energy Calculator - GPU Acceleration Example" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        compare_cpu_gpu_performance();
        demonstrate_gpu_memory_management();
        demonstrate_optimization_techniques();
        
        std::cout << "\n✓ All GPU acceleration examples completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}