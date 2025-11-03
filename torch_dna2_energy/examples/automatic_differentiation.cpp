#include "dna2_energy_calculator.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace dna2;

/**
 * @file automatic_differentiation.cpp
 * @brief Example demonstrating automatic differentiation capabilities
 * 
 * This example shows how to:
 * - Use automatic differentiation for gradient computation
 * - Perform energy minimization using gradients
 * - Analyze force fields and energy landscapes
 * - Use gradients for structural optimization
 */

void create_test_dna_system(DNAParticle& particles, int strand_length = 10) {
    const int N = particles.num_particles;
    
    // Create a slightly perturbed DNA duplex
    for (int i = 0; i < N; ++i) {
        float x = (i % strand_length) * 0.7f;
        float y = (i < strand_length) ? 0.0f : 2.0f;
        float z = 0.0f;
        
        // Add small random perturbation
        x += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        y += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        z += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        
        particles.positions[i][0] = x;
        particles.positions[i][1] = y;
        particles.positions[i][2] = z;
        
        particles.strand_ids[i] = (i < strand_length) ? 0 : 1;
        
        // Complementary base pairing
        int base_idx = i % strand_length;
        if (base_idx % 2 == 0) {
            particles.types[i] = (i < strand_length) ? 0 : 1;  // A-T
            particles.btypes[i] = (i < strand_length) ? 0 : 1;
        } else {
            particles.types[i] = (i < strand_length) ? 2 : 3;  // G-C
            particles.btypes[i] = (i < strand_length) ? 2 : 3;
        }
        
        // Backbone bonding
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
        
        // Slightly varied orientations
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
        float angle = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        particles.orientations[i][0][0] = cos_a;
        particles.orientations[i][0][2] = sin_a;
        particles.orientations[i][2][0] = -sin_a;
        particles.orientations[i][2][2] = cos_a;
    }
    
    particles.compute_interaction_centers();
}

void demonstrate_basic_autograd() {
    std::cout << "\n=== Basic Automatic Differentiation Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 10;
    const int N = 2 * strand_length;
    
    // Create test system
    DNAParticle particles(N, torch::kCPU);
    create_test_dna_system(particles, strand_length);
    
    std::cout << "Created DNA system with " << N << " nucleotides" << std::endl;
    
    // Enable gradients for positions and orientations
    auto positions_grad = particles.positions.clone().detach().set_requires_grad(true);
    auto orientations_grad = particles.orientations.clone().detach().set_requires_grad(true);
    
    std::cout << "✓ Gradients enabled for positions and orientations" << std::endl;
    
    // Compute energy with automatic differentiation
    std::cout << "\nComputing energy with automatic differentiation..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    torch::Tensor energy = calculator.compute_energy_autograd(positions_grad, orientations_grad);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Energy: " << std::fixed << std::setprecision(6) << energy.item<float>() << " kT" << std::endl;
    std::cout << "Computation time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Requires gradient: " << (energy.requires_grad() ? "Yes" : "No") << std::endl;
    
    // Compute gradients
    std::cout << "\nComputing gradients..." << std::endl;
    energy.backward();
    
    torch::Tensor position_gradients = positions_grad.grad();
    torch::Tensor orientation_gradients = orientations_grad.grad();
    
    std::cout << "✓ Gradients computed successfully" << std::endl;
    std::cout << "Position gradients shape: " << position_gradients.sizes() << std::endl;
    std::cout << "Orientation gradients shape: " << orientation_gradients.sizes() << std::endl;
    
    // Analyze gradient statistics
    float pos_grad_max = torch::max(torch::abs(position_gradients)).item<float>();
    float pos_grad_mean = torch::mean(torch::abs(position_gradients)).item<float>();
    float orient_grad_max = torch::max(torch::abs(orientation_gradients)).item<float>();
    float orient_grad_mean = torch::mean(torch::abs(orientation_gradients)).item<float>();
    
    std::cout << "\nGradient statistics:" << std::endl;
    std::cout << "Position gradients - Max: " << std::scientific << pos_grad_max 
              << ", Mean: " << pos_grad_mean << std::endl;
    std::cout << "Orientation gradients - Max: " << std::scientific << orient_grad_max 
              << ", Mean: " << orient_grad_mean << std::endl;
    
    // Display some sample gradients
    std::cout << "\nSample position gradients:" << std::endl;
    for (int i = 0; i < std::min(3, N); ++i) {
        std::cout << "  Particle " << i << ": [" 
                  << std::scientific << std::setprecision(3)
                  << position_gradients[i][0].item<float>() << ", "
                  << position_gradients[i][1].item<float>() << ", "
                  << position_gradients[i][2].item<float>() << "]" << std::endl;
    }
}

void demonstrate_gradient_descent_optimization() {
    std::cout << "\n=== Gradient Descent Optimization Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 8;
    const int N = 2 * strand_length;
    
    // Create initial system with larger perturbations
    DNAParticle particles(N, torch::kCPU);
    create_test_dna_system(particles, strand_length);
    
    // Add larger perturbations for optimization demo
    particles.positions += torch::randn_like(particles.positions) * 0.5f;
    particles.compute_interaction_centers();
    
    std::cout << "Starting optimization with " << N << " nucleotides" << std::endl;
    
    // Initialize variables for optimization
    auto positions_opt = particles.positions.clone().detach().set_requires_grad(true);
    auto orientations_opt = particles.orientations.clone().detach().set_requires_grad(true);
    
    // Optimization parameters
    float learning_rate = 0.01f;
    int max_iterations = 100;
    float tolerance = 1e-6f;
    
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Max iterations: " << max_iterations << std::endl;
    
    // Gradient descent optimization
    std::vector<float> energy_history;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Zero gradients
        if (positions_opt.grad().defined()) {
            positions_opt.grad().zero_();
        }
        if (orientations_opt.grad().defined()) {
            orientations_opt.grad().zero_();
        }
        
        // Compute energy and gradients
        torch::Tensor energy = calculator.compute_energy_autograd(positions_opt, orientations_opt);
        energy.backward();
        
        // Store energy history
        energy_history.push_back(energy.item<float>());
        
        // Check convergence
        if (iter > 0 && std::abs(energy_history[iter] - energy_history[iter-1]) < tolerance) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
        
        // Update positions (gradient descent)
        torch::NoGradGuard no_grad;
        positions_opt -= learning_rate * positions_opt.grad();
        
        // Update orientations (more careful update for rotation matrices)
        torch::Tensor orient_update = learning_rate * orientations_opt.grad();
        orientations_opt -= orient_update;
        
        // Re-orthonormalize rotation matrices
        for (int i = 0; i < N; ++i) {
            // Simple Gram-Schmidt orthonormalization
            torch::Tensor r = orientations_opt[i];
            torch::Tensor c1 = r[0] / torch::norm(r[0]);
            torch::Tensor c2 = r[1] - torch::dot(r[1], c1) * c1;
            c2 = c2 / torch::norm(c2);
            torch::Tensor c3 = torch::cross(c1, c2);
            orientations_opt[i][0] = c1;
            orientations_opt[i][1] = c2;
            orientations_opt[i][2] = c3;
        }
        
        // Progress update
        if (iter % 10 == 0 || iter == max_iterations - 1) {
            std::cout << "Iteration " << std::setw(3) << iter 
                      << ": Energy = " << std::fixed << std::setprecision(6) 
                      << energy.item<float>() << " kT" << std::endl;
        }
    }
    
    std::cout << "\nOptimization completed!" << std::endl;
    std::cout << "Initial energy: " << std::fixed << std::setprecision(6) 
              << energy_history[0] << " kT" << std::endl;
    std::cout << "Final energy: " << std::setprecision(6) 
              << energy_history.back() << " kT" << std::endl;
    std::cout << "Energy reduction: " << std::setprecision(6) 
              << (energy_history[0] - energy_history.back()) << " kT" << std::endl;
}

void demonstrate_force_field_analysis() {
    std::cout << "\n=== Force Field Analysis Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int N = 20;
    
    // Create simple linear chain
    DNAParticle particles(N, torch::kCPU);
    for (int i = 0; i < N; ++i) {
        particles.positions[i][0] = i * 0.7f;
        particles.positions[i][1] = 0.0f;
        particles.positions[i][2] = 0.0f;
        particles.types[i] = i % 4;
        particles.btypes[i] = i % 4;
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
    
    std::cout << "Created linear chain of " << N << " nucleotides" << std::endl;
    
    // Compare analytical forces with autograd forces
    std::cout << "\nComparing analytical and autograd forces..." << std::endl;
    
    // Analytical forces
    torch::Tensor analytical_forces = calculator.compute_forces(particles);
    
    // Autograd forces
    auto positions_grad = particles.positions.clone().detach().set_requires_grad(true);
    auto orientations_grad = particles.orientations.clone().detach().set_requires_grad(true);
    
    torch::Tensor energy = calculator.compute_energy_autograd(positions_grad, orientations_grad);
    energy.backward();
    torch::Tensor autograd_forces = -positions_grad.grad();  // Forces are negative gradients
    
    // Compare forces
    torch::Tensor force_diff = torch::norm(analytical_forces - autograd_forces);
    torch::Tensor max_diff = torch::max(torch::abs(analytical_forces - autograd_forces));
    
    std::cout << "Force difference (L2 norm): " << std::scientific << force_diff.item<float>() << std::endl;
    std::cout << "Maximum force difference: " << max_diff.item<float>() << std::endl;
    
    if (force_diff.item<float>() < 1e-5) {
        std::cout << "✓ Analytical and autograd forces are consistent" << std::endl;
    } else {
        std::cout << "✗ Significant differences detected between force calculations" << std::endl;
    }
    
    // Analyze force components
    std::cout << "\nForce analysis for selected particles:" << std::endl;
    std::cout << std::setw(8) << "Particle" << std::setw(12) << "|F_analytical|" 
              << std::setw(12) << "|F_autograd|" << std::setw(12) << "Difference" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    
    for (int i = 0; i < std::min(5, N); ++i) {
        float f_analytical = torch::norm(analytical_forces[i]).item<float>();
        float f_autograd = torch::norm(autograd_forces[i]).item<float>();
        float diff = std::abs(f_analytical - f_autograd);
        
        std::cout << std::setw(8) << i
                  << std::setw(12) << std::fixed << std::setprecision(6) << f_analytical
                  << std::setw(12) << std::setprecision(6) << f_autograd
                  << std::setw(12) << std::scientific << diff << std::endl;
    }
}

void demonstrate_energy_landscape() {
    std::cout << "\n=== Energy Landscape Exploration Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int N = 2;  // Simple 2-particle system for landscape analysis
    
    // Create two nucleotides
    DNAParticle particles(N, torch::kCPU);
    particles.positions[0] = torch::tensor({0.0f, 0.0f, 0.0f});
    particles.positions[1] = torch::tensor({1.0f, 0.0f, 0.0f});
    particles.types[0] = 0;  // A
    particles.types[1] = 1;  // T
    particles.btypes[0] = 0;
    particles.btypes[1] = 1;
    particles.strand_ids[0] = 0;
    particles.strand_ids[1] = 1;
    particles.orientations[0] = torch::eye(3, torch::kFloat32);
    particles.orientations[1] = torch::eye(3, torch::kFloat32);
    particles.compute_interaction_centers();
    
    std::cout << "Created 2-particle system for energy landscape analysis" << std::endl;
    
    // Explore energy as function of separation
    std::cout << "\nEnergy vs separation distance:" << std::endl;
    std::cout << std::setw(10) << "Distance" << std::setw(15) << "Energy" 
              << std::setw(15) << "Force" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    std::vector<float> distances;
    std::vector<float> energies;
    std::vector<float> forces;
    
    for (float d = 0.5f; d <= 3.0f; d += 0.1f) {
        // Update particle separation
        particles.positions[1][0] = d;
        particles.compute_interaction_centers();
        
        // Compute energy and force
        torch::Tensor energy = calculator.compute_energy(particles);
        torch::Tensor force_tensor = calculator.compute_forces(particles);
        float force = torch::norm(force_tensor[1] - force_tensor[0]).item<float>();
        
        distances.push_back(d);
        energies.push_back(energy.item<float>());
        forces.push_back(force);
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << d
                  << std::setw(15) << std::setprecision(6) << energy.item<float>()
                  << std::setw(15) << std::setprecision(6) << force << std::endl;
    }
    
    // Find minimum energy distance
    auto min_it = std::min_element(energies.begin(), energies.end());
    int min_idx = std::distance(energies.begin(), min_it);
    
    std::cout << "\nMinimum energy found at distance: " << std::fixed << std::setprecision(2) 
              << distances[min_idx] << " nm" << std::endl;
    std::cout << "Minimum energy: " << std::setprecision(6) << energies[min_idx] << " kT" << std::endl;
    
    // Verify with gradient at minimum
    particles.positions[1][0] = distances[min_idx];
    particles.compute_interaction_centers();
    
    auto positions_grad = particles.positions.clone().detach().set_requires_grad(true);
    auto orientations_grad = particles.orientations.clone().detach().set_requires_grad(true);
    
    torch::Tensor energy_min = calculator.compute_energy_autograd(positions_grad, orientations_grad);
    energy_min.backward();
    
    torch::Tensor gradient = positions_grad.grad();
    float grad_magnitude = torch::norm(gradient).item<float>();
    
    std::cout << "Gradient magnitude at minimum: " << std::scientific << grad_magnitude << std::endl;
    
    if (grad_magnitude < 1e-4) {
        std::cout << "✓ Gradient near zero at energy minimum (as expected)" << std::endl;
    } else {
        std::cout << "⚠ Gradient not zero at minimum - may need finer resolution" << std::endl;
    }
}

int main() {
    std::cout << "DNA2 Energy Calculator - Automatic Differentiation Example" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try {
        demonstrate_basic_autograd();
        demonstrate_gradient_descent_optimization();
        demonstrate_force_field_analysis();
        demonstrate_energy_landscape();
        
        std::cout << "\n✓ All automatic differentiation examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}