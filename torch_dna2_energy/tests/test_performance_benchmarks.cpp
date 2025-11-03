#include "dna2_energy_calculator.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <fstream>

using namespace dna2;

/**
 * @file test_performance_benchmarks.cpp
 * @brief Performance benchmarks comparing CPU vs GPU performance
 * 
 * This test suite provides:
 * - CPU vs GPU performance comparison
 * - Scaling analysis with system size
 * - Batch processing performance
 * - Memory usage analysis
 */

class PerformanceBenchmarks {
private:
    struct BenchmarkResult {
        std::string test_name;
        int system_size;
        float cpu_time_us;
        float gpu_time_us;
        float speedup;
        float cpu_energy;
        float gpu_energy;
        float energy_diff;
    };
    
    std::vector<BenchmarkResult> results;
    
    void create_test_system(DNAParticle& particles, int num_strands, int strand_length) {
        const int N = particles.num_particles;
        
        // Create multiple strands arranged in a grid
        int grid_size = static_cast<int>(std::sqrt(num_strands)) + 1;
        float spacing = 3.0f;
        
        for (int i = 0; i < N; ++i) {
            int strand_id = i / strand_length;
            int pos_in_strand = i % strand_length;
            
            int grid_x = strand_id % grid_size;
            int grid_y = strand_id / grid_size;
            
            float x = pos_in_strand * 0.7f + grid_x * spacing * strand_length * 0.7f;
            float y = grid_y * spacing;
            float z = 0.0f;
            
            particles.positions[i][0] = x;
            particles.positions[i][1] = y;
            particles.positions[i][2] = z;
            
            particles.strand_ids[i] = strand_id;
            particles.types[i] = pos_in_strand % 4;
            particles.btypes[i] = pos_in_strand % 4;
            
            if (pos_in_strand < strand_length - 1) {
                particles.n3_neighbors[i] = i + 1;
                particles.n5_neighbors[i + 1] = i;
                particles.bonded_mask[i][i + 1] = true;
                particles.bonded_mask[i + 1][i] = true;
            }
            
            particles.orientations[i] = torch::eye(3, torch::kFloat32);
        }
        
        particles.compute_interaction_centers();
    }
    
    float time_energy_computation(DNA2EnergyCalculator& calculator, 
                                 const DNAParticle& particles, int num_runs = 10) {
        // Warm up
        calculator.compute_energy(particles);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            calculator.compute_energy(particles);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<float>(duration.count()) / num_runs;
    }
    
    float time_force_computation(DNA2EnergyCalculator& calculator, 
                                const DNAParticle& particles, int num_runs = 10) {
        // Warm up
        calculator.compute_forces(particles);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            calculator.compute_forces(particles);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<float>(duration.count()) / num_runs;
    }
    
public:
    void run_scaling_benchmarks() {
        std::cout << "\n=== Scaling Benchmarks ===" << std::endl;
        
        std::vector<std::pair<int, int>> test_sizes = {
            {2, 10},    // 20 particles
            {4, 15},    // 60 particles
            {8, 20},    // 160 particles
            {16, 25},   // 400 particles
            {32, 25},   // 800 particles
            {64, 25}    // 1600 particles
        };
        
        std::cout << std::setw(12) << "Particles" << std::setw(15) << "CPU Energy (μs)" 
                  << std::setw(15) << "CPU Forces (μs)" << std::setw(15) << "GPU Energy (μs)" 
                  << std::setw(15) << "GPU Forces (μs)" << std::setw(12) << "Energy Speedup" 
                  << std::setw(12) << "Force Speedup" << std::endl;
        std::cout << std::string(96, '-') << std::endl;
        
        for (const auto& size : test_sizes) {
            int num_strands = size.first;
            int strand_length = size.second;
            int N = num_strands * strand_length;
            
            // Create test system
            DNAParticle particles(N, torch::kCPU);
            create_test_system(particles, num_strands, strand_length);
            
            // CPU benchmarks
            DNA2EnergyCalculator cpu_calculator;
            float cpu_energy_time = time_energy_computation(cpu_calculator, particles);
            float cpu_force_time = time_force_computation(cpu_calculator, particles);
            torch::Tensor cpu_energy = cpu_calculator.compute_energy(particles);
            
            // GPU benchmarks (if available)
            float gpu_energy_time = -1.0f;
            float gpu_force_time = -1.0f;
            float energy_speedup = 0.0f;
            float force_speedup = 0.0f;
            torch::Tensor gpu_energy;
            
            try {
                auto gpu_calculator = create_gpu_calculator();
                
                DNAParticle gpu_particles = particles;
                gpu_particles.to(torch::kCUDA);
                
                gpu_energy_time = time_energy_computation(gpu_calculator, gpu_particles);
                gpu_force_time = time_force_computation(gpu_calculator, gpu_particles);
                gpu_energy = gpu_calculator.compute_energy(gpu_particles);
                
                energy_speedup = cpu_energy_time / gpu_energy_time;
                force_speedup = cpu_force_time / gpu_force_time;
                
            } catch (const std::exception& e) {
                std::cout << "GPU not available: " << e.what() << std::endl;
            }
            
            // Store results
            BenchmarkResult result;
            result.test_name = "Scaling";
            result.system_size = N;
            result.cpu_time_us = cpu_energy_time;
            result.gpu_time_us = gpu_energy_time;
            result.speedup = energy_speedup;
            result.cpu_energy = cpu_energy.item<float>();
            result.gpu_energy = gpu_energy.numel() > 0 ? gpu_energy.cpu().item<float>() : -1.0f;
            result.energy_diff = (result.gpu_energy > 0) ? 
                                std::abs(result.cpu_energy - result.gpu_energy) : 0.0f;
            results.push_back(result);
            
            // Display results
            std::cout << std::setw(12) << N
                      << std::setw(15) << std::fixed << std::setprecision(2) << cpu_energy_time
                      << std::setw(15) << std::setprecision(2) << cpu_force_time;
            
            if (gpu_energy_time > 0) {
                std::cout << std::setw(15) << std::setprecision(2) << gpu_energy_time
                          << std::setw(15) << std::setprecision(2) << gpu_force_time
                          << std::setw(12) << std::setprecision(2) << energy_speedup
                          << std::setw(12) << std::setprecision(2) << force_speedup;
            } else {
                std::cout << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(12) << "N/A"
                          << std::setw(12) << "N/A";
            }
            
            std::cout << std::endl;
        }
    }
    
    void run_batch_benchmarks() {
        std::cout << "\n=== Batch Processing Benchmarks ===" << std::endl;
        
        const int strand_length = 20;
        const int N = 2 * strand_length;
        
        std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64};
        
        std::cout << std::setw(12) << "Batch Size" << std::setw(15) << "Individual (μs)" 
                  << std::setw(15) << "Batch (μs)" << std::setw(15) << "Per Item (μs)" 
                  << std::setw(12) << "Speedup" << std::setw(15) << "GPU Batch (μs)" 
                  << std::setw(12) << "GPU Speedup" << std::endl;
        std::cout << std::string(96, '-') << std::endl;
        
        // Create base system
        DNAParticle base_particles(N, torch::kCPU);
        create_test_system(base_particles, 2, strand_length);
        
        DNA2EnergyCalculator cpu_calculator;
        
        for (int batch_size : batch_sizes) {
            // Create batch
            std::vector<DNAParticle> batch;
            for (int i = 0; i < batch_size; ++i) {
                DNAParticle particles = base_particles;
                particles.positions += torch::rand_like(particles.positions) * 0.1f;
                particles.compute_interaction_centers();
                batch.push_back(particles);
            }
            
            // Individual processing time
            auto start = std::chrono::high_resolution_clock::now();
            for (const auto& particles : batch) {
                cpu_calculator.compute_energy(particles);
            }
            auto end = std::chrono::high_resolution_clock::now();
            float individual_time = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            
            // Batch processing time
            start = std::chrono::high_resolution_clock::now();
            torch::Tensor batch_energy = cpu_calculator.compute_energy_batch(batch);
            end = std::chrono::high_resolution_clock::now();
            float batch_time = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            
            float per_item_time = batch_time / batch_size;
            float speedup = individual_time / batch_time;
            
            // GPU batch processing (if available)
            float gpu_batch_time = -1.0f;
            float gpu_speedup = 0.0f;
            
            try {
                auto gpu_calculator = create_gpu_calculator();
                
                std::vector<DNAParticle> gpu_batch;
                for (const auto& particles : batch) {
                    DNAParticle gpu_particles = particles;
                    gpu_particles.to(torch::kCUDA);
                    gpu_batch.push_back(gpu_particles);
                }
                
                start = std::chrono::high_resolution_clock::now();
                torch::Tensor gpu_batch_energy = gpu_calculator.compute_energy_batch(gpu_batch);
                end = std::chrono::high_resolution_clock::now();
                gpu_batch_time = static_cast<float>(
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                
                gpu_speedup = individual_time / gpu_batch_time;
                
            } catch (const std::exception& e) {
                // GPU not available
            }
            
            // Display results
            std::cout << std::setw(12) << batch_size
                      << std::setw(15) << std::fixed << std::setprecision(2) << individual_time
                      << std::setw(15) << std::setprecision(2) << batch_time
                      << std::setw(15) << std::setprecision(2) << per_item_time
                      << std::setw(12) << std::setprecision(2) << speedup;
            
            if (gpu_batch_time > 0) {
                std::cout << std::setw(15) << std::setprecision(2) << gpu_batch_time
                          << std::setw(12) << std::setprecision(2) << gpu_speedup;
            } else {
                std::cout << std::setw(15) << "N/A"
                          << std::setw(12) << "N/A";
            }
            
            std::cout << std::endl;
        }
    }
    
    void run_memory_benchmarks() {
        std::cout << "\n=== Memory Usage Benchmarks ===" << std::endl;
        
        std::vector<int> system_sizes = {100, 200, 400, 800, 1600};
        
        std::cout << std::setw(12) << "Particles" << std::setw(15) << "Memory (MB)" 
                  << std::setw(15) << "Per Particle (KB)" << std::setw(15) << "GPU Memory (MB)" 
                  << std::setw(15) << "GPU Per Particle (KB)" << std::endl;
        std::cout << std::string(72, '-') << std::endl;
        
        for (int N : system_sizes) {
            DNAParticle particles(N, torch::kCPU);
            create_test_system(particles, N / 20, 20);
            
            // Estimate memory usage
            size_t cpu_memory = 0;
            cpu_memory += particles.positions.numel() * sizeof(float);
            cpu_memory += particles.orientations.numel() * sizeof(float);
            cpu_memory += particles.backbone_centers.numel() * sizeof(float);
            cpu_memory += particles.stack_centers.numel() * sizeof(float);
            cpu_memory += particles.base_centers.numel() * sizeof(float);
            cpu_memory += particles.types.numel() * sizeof(int);
            cpu_memory += particles.btypes.numel() * sizeof(int);
            cpu_memory += particles.strand_ids.numel() * sizeof(int);
            cpu_memory += particles.n3_neighbors.numel() * sizeof(int);
            cpu_memory += particles.n5_neighbors.numel() * sizeof(int);
            cpu_memory += particles.bonded_mask.numel() * sizeof(bool);
            
            float cpu_memory_mb = cpu_memory / (1024.0f * 1024.0f);
            float cpu_per_particle_kb = (cpu_memory / N) / 1024.0f;
            
            float gpu_memory_mb = -1.0f;
            float gpu_per_particle_kb = -1.0f;
            
            try {
                DNAParticle gpu_particles = particles;
                gpu_particles.to(torch::kCUDA);
                
                // GPU memory usage should be similar to CPU
                gpu_memory_mb = cpu_memory_mb;
                gpu_per_particle_kb = cpu_per_particle_kb;
                
            } catch (const std::exception& e) {
                // GPU not available
            }
            
            std::cout << std::setw(12) << N
                      << std::setw(15) << std::fixed << std::setprecision(2) << cpu_memory_mb
                      << std::setw(15) << std::setprecision(2) << cpu_per_particle_kb;
            
            if (gpu_memory_mb > 0) {
                std::cout << std::setw(15) << std::setprecision(2) << gpu_memory_mb
                          << std::setw(15) << std::setprecision(2) << gpu_per_particle_kb;
            } else {
                std::cout << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A";
            }
            
            std::cout << std::endl;
        }
    }
    
    void run_autograd_benchmarks() {
        std::cout << "\n=== Automatic Differentiation Benchmarks ===" << std::endl;
        
        std::vector<int> system_sizes = {50, 100, 200, 400};
        
        std::cout << std::setw(12) << "Particles" << std::setw(15) << "Forward (μs)" 
                  << std::setw(15) << "Autograd (μs)" << std::setw(15) << "Backward (μs)" 
                  << std::setw(12) << "Overhead" << std::setw(15) << "GPU Autograd (μs)" 
                  << std::setw(12) << "GPU Speedup" << std::endl;
        std::cout << std::string(96, '-') << std::endl;
        
        for (int N : system_sizes) {
            DNAParticle particles(N, torch::kCPU);
            create_test_system(particles, N / 20, 20);
            
            DNA2EnergyCalculator calculator;
            
            // Forward pass time
            float forward_time = time_energy_computation(calculator, particles);
            
            // Autograd time
            auto positions_grad = particles.positions.clone().detach().set_requires_grad(true);
            auto orientations_grad = particles.orientations.clone().detach().set_requires_grad(true);
            
            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor energy_grad = calculator.compute_energy_autograd(positions_grad, orientations_grad);
            auto end = std::chrono::high_resolution_clock::now();
            float autograd_time = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            
            // Backward pass time
            start = std::chrono::high_resolution_clock::now();
            energy_grad.backward();
            end = std::chrono::high_resolution_clock::now();
            float backward_time = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            
            float overhead = autograd_time / forward_time;
            
            // GPU autograd (if available)
            float gpu_autograd_time = -1.0f;
            float gpu_speedup = 0.0f;
            
            try {
                auto gpu_calculator = create_gpu_calculator();
                
                DNAParticle gpu_particles = particles;
                gpu_particles.to(torch::kCUDA);
                
                auto gpu_positions_grad = gpu_particles.positions.clone().detach().set_requires_grad(true);
                auto gpu_orientations_grad = gpu_particles.orientations.clone().detach().set_requires_grad(true);
                
                start = std::chrono::high_resolution_clock::now();
                torch::Tensor gpu_energy_grad = gpu_calculator.compute_energy_autograd(
                    gpu_positions_grad, gpu_orientations_grad);
                end = std::chrono::high_resolution_clock::now();
                gpu_autograd_time = static_cast<float>(
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                
                gpu_speedup = autograd_time / gpu_autograd_time;
                
            } catch (const std::exception& e) {
                // GPU not available
            }
            
            // Display results
            std::cout << std::setw(12) << N
                      << std::setw(15) << std::fixed << std::setprecision(2) << forward_time
                      << std::setw(15) << std::setprecision(2) << autograd_time
                      << std::setw(15) << std::setprecision(2) << backward_time
                      << std::setw(12) << std::setprecision(2) << overhead;
            
            if (gpu_autograd_time > 0) {
                std::cout << std::setw(15) << std::setprecision(2) << gpu_autograd_time
                          << std::setw(12) << std::setprecision(2) << gpu_speedup;
            } else {
                std::cout << std::setw(15) << "N/A"
                          << std::setw(12) << "N/A";
            }
            
            std::cout << std::endl;
        }
    }
    
    void save_results(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Could not open file for writing: " << filename << std::endl;
            return;
        }
        
        file << "Test Name,System Size,CPU Time (μs),GPU Time (μs),Speedup,CPU Energy,GPU Energy,Energy Diff\n";
        
        for (const auto& result : results) {
            file << result.test_name << ","
                 << result.system_size << ","
                 << result.cpu_time_us << ","
                 << result.gpu_time_us << ","
                 << result.speedup << ","
                 << result.cpu_energy << ","
                 << result.gpu_energy << ","
                 << result.energy_diff << "\n";
        }
        
        file.close();
        std::cout << "\nResults saved to: " << filename << std::endl;
    }
    
    void run_all_benchmarks() {
        std::cout << "DNA2 Performance Benchmarks" << std::endl;
        std::cout << "===========================" << std::endl;
        
        run_scaling_benchmarks();
        run_batch_benchmarks();
        run_memory_benchmarks();
        run_autograd_benchmarks();
        
        save_results("benchmark_results.csv");
        
        std::cout << "\n✓ All benchmarks completed!" << std::endl;
    }
};

int main() {
    try {
        PerformanceBenchmarks benchmarks;
        benchmarks.run_all_benchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with exception: " << e.what() << std::endl;
        return 1;
    }
}