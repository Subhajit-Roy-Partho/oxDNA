#include "dna2_energy_calculator.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dna2;

void test_data_structures() {
    std::cout << "Testing data structures..." << std::endl;
    
    // Test DNA2Parameters
    DNA2Parameters params;
    params.initialize_tensors();
    assert(params.excl_sigma.sizes() == std::vector<int64_t>{4});
    assert(params.hydr_eps_matrix.sizes() == std::vector<int64_t>{5, 5});
    
    // Test DNAParticle
    const int N = 10;
    DNAParticle particles(N, torch::kCPU);
    assert(particles.positions.sizes() == std::vector<int64_t>{N, 3});
    assert(particles.orientations.sizes() == std::vector<int64_t>{N, 3, 3});
    assert(particles.num_particles == N);
    
    // Test InteractionResult
    InteractionResult result(N, 1, torch::kCPU);
    assert(result.total_energy.sizes() == std::vector<int64_t>{1});
    assert(result.forces.sizes() == std::vector<int64_t>{1, N, 3});
    
    std::cout << "✓ Data structures test passed" << std::endl;
}

void test_mathematical_functions() {
    std::cout << "Testing mathematical functions..." << std::endl;
    
    DNA2Parameters params;
    params.initialize_tensors();
    
    // Test tensor operations
    torch::Tensor a = torch::tensor({1.0f, 0.0f, 0.0f});
    torch::Tensor b = torch::tensor({0.0f, 1.0f, 0.0f});
    
    auto cross = TensorOperations::cross_product(a, b);
    auto expected = torch::tensor({0.0f, 0.0f, 1.0f});
    assert(torch::allclose(cross, expected, 1e-6));
    
    auto dot = TensorOperations::dot_product(a, b);
    assert(std::abs(dot.item<float>() - 0.0f) < 1e-6);
    
    auto norm = TensorOperations::normalize(a);
    assert(torch::allclose(norm, a, 1e-6));
    
    std::cout << "✓ Mathematical functions test passed" << std::endl;
}

void test_mesh_interpolation() {
    std::cout << "Testing mesh interpolation..." << std::endl;
    
    DNA2Parameters params;
    params.initialize_tensors();
    
    auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
    assert(interpolator->is_initialized());
    
    // Test interpolation
    torch::Tensor cos_values = torch::linspace(-1.0, 1.0, 10);
    auto interpolated = interpolator->interpolate(MeshType::HYDR_F4_THETA1, cos_values);
    assert(interpolated.sizes() == cos_values.sizes());
    
    auto derivative = interpolator->interpolate_derivative(MeshType::HYDR_F4_THETA1, cos_values);
    assert(derivative.sizes() == cos_values.sizes());
    
    std::cout << "✓ Mesh interpolation test passed" << std::endl;
}

void test_energy_calculator() {
    std::cout << "Testing energy calculator..." << std::endl;
    
    // Create calculator
    DNA2EnergyCalculator calculator;
    
    // Create simple test system
    const int N = 4;
    DNAParticle particles(N, torch::kCPU);
    
    // Set up simple configuration
    for (int i = 0; i < N; ++i) {
        particles.positions[i][0] = i * 0.7f;
        particles.positions[i][1] = 0.0f;
        particles.positions[i][2] = 0.0f;
        particles.types[i] = i % 2;
        particles.btypes[i] = i % 2;
        particles.strand_ids[i] = 0;
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
        
        if (i > 0) {
            particles.n3_neighbors[i-1] = i;
            particles.n5_neighbors[i] = i-1;
            particles.bonded_mask[i-1][i] = true;
            particles.bonded_mask[i][i-1] = true;
        }
    }
    
    particles.compute_interaction_centers();
    
    // Test energy computation
    torch::Tensor energy = calculator.compute_energy(particles);
    assert(energy.numel() == 1);
    assert(std::isfinite(energy.item<float>()));
    
    // Test force computation
    torch::Tensor forces = calculator.compute_forces(particles);
    assert(forces.sizes() == std::vector<int64_t>{N, 3});
    
    // Test joint computation
    auto [energy2, forces2] = calculator.compute_energy_and_forces(particles);
    assert(torch::allclose(energy, energy2, 1e-6));
    assert(torch::allclose(forces, forces2, 1e-6));
    
    std::cout << "✓ Energy calculator test passed" << std::endl;
}

void test_batch_processing() {
    std::cout << "Testing batch processing..." << std::endl;
    
    DNA2EnergyCalculator calculator;
    
    // Create test systems
    const int N = 4;
    std::vector<DNAParticle> batch;
    
    for (int b = 0; b < 3; ++b) {
        DNAParticle particles(N, torch::kCPU);
        
        for (int i = 0; i < N; ++i) {
            particles.positions[i][0] = i * 0.7f + b * 0.1f;
            particles.positions[i][1] = 0.0f;
            particles.positions[i][2] = 0.0f;
            particles.types[i] = i % 2;
            particles.btypes[i] = i % 2;
            particles.strand_ids[i] = 0;
            particles.orientations[i] = torch::eye(3, torch::kFloat32);
            
            if (i > 0) {
                particles.n3_neighbors[i-1] = i;
                particles.n5_neighbors[i] = i;
                particles.bonded_mask[i-1][i] = true;
                particles.bonded_mask[i][i-1] = true;
            }
        }
        
        particles.compute_interaction_centers();
        batch.push_back(particles);
    }
    
    // Test batch energy computation
    torch::Tensor batch_energy = calculator.compute_energy_batch(batch);
    assert(batch_energy.sizes() == std::vector<int64_t>{3});
    
    // Test batch force computation
    torch::Tensor batch_forces = calculator.compute_forces_batch(batch);
    assert(batch_forces.sizes() == std::vector<int64_t>{3, N, 3});
    
    std::cout << "✓ Batch processing test passed" << std::endl;
}

void test_automatic_differentiation() {
    std::cout << "Testing automatic differentiation..." << std::endl;
    
    DNA2EnergyCalculator calculator;
    
    // Create test system
    const int N = 4;
    DNAParticle particles(N, torch::kCPU);
    
    for (int i = 0; i < N; ++i) {
        particles.positions[i][0] = i * 0.7f;
        particles.positions[i][1] = 0.0f;
        particles.positions[i][2] = 0.0f;
        particles.types[i] = i % 2;
        particles.btypes[i] = i % 2;
        particles.strand_ids[i] = 0;
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
        
        if (i > 0) {
            particles.n3_neighbors[i-1] = i;
            particles.n5_neighbors[i] = i;
            particles.bonded_mask[i-1][i] = true;
            particles.bonded_mask[i][i-1] = true;
        }
    }
    
    particles.compute_interaction_centers();
    
    // Test autograd
    auto positions_grad = particles.positions.clone().detach().set_requires_grad(true);
    auto orientations_grad = particles.orientations.clone().detach().set_requires_grad(true);
    
    torch::Tensor energy_grad = calculator.compute_energy_autograd(positions_grad, orientations_grad);
    assert(energy_grad.requires_grad());
    
    // Compute gradients
    energy_grad.backward();
    torch::Tensor position_gradients = positions_grad.grad();
    assert(position_gradients.sizes() == particles.positions.sizes());
    assert(torch::all(torch::isfinite(position_gradients)));
    
    std::cout << "✓ Automatic differentiation test passed" << std::endl;
}

void run_performance_test() {
    std::cout << "Running performance test..." << std::endl;
    
    DNA2EnergyCalculator calculator;
    
    // Create larger test system
    const int N = 100;
    DNAParticle particles(N, torch::kCPU);
    
    for (int i = 0; i < N; ++i) {
        particles.positions[i][0] = (i % 10) * 0.7f;
        particles.positions[i][1] = (i / 10) * 0.7f;
        particles.positions[i][2] = 0.0f;
        particles.types[i] = i % 4;
        particles.btypes[i] = i % 4;
        particles.strand_ids[i] = i / 10;
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
        
        if (i % 10 != 9) {
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        }
    }
    
    particles.compute_interaction_centers();
    
    // Time energy computation
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor energy = calculator.compute_energy(particles);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Energy computation for " << N << " particles: " 
              << duration.count() << " microseconds" << std::endl;
    
    // Time force computation
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor forces = calculator.compute_forces(particles);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Force computation for " << N << " particles: " 
              << duration.count() << " microseconds" << std::endl;
    
    std::cout << "✓ Performance test completed" << std::endl;
}

int main() {
    std::cout << "DNA2 Energy Calculator Basic Tests" << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        test_data_structures();
        test_mathematical_functions();
        test_mesh_interpolation();
        test_energy_calculator();
        test_batch_processing();
        test_automatic_differentiation();
        run_performance_test();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}