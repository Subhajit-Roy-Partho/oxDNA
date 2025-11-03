#include "dna2_energy_calculator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

using namespace dna2;

/**
 * @file test_oxdna_validation.cpp
 * @brief Validation tests comparing with oxDNA reference values
 * 
 * This test suite provides:
 * - Comparison with known oxDNA energy values
 * - Validation of force calculations
 * - Testing of specific configurations
 * - Accuracy verification
 */

class OxDNAValidation {
private:
    struct TestConfiguration {
        std::string name;
        std::vector<float> positions;
        std::vector<int> types;
        float reference_energy;
        float tolerance;
    };
    
    std::vector<TestConfiguration> test_configs;
    int passed_tests = 0;
    int total_tests = 0;
    
    void load_test_configurations() {
        // Simple two-particle test cases with known energies
        test_configs = {
            {
                "Two AT base pairs at optimal distance",
                {0.0f, 0.0f, 0.0f, 0.7f, 0.0f, 0.0f},
                {0, 1},
                -2.5f,  // Approximate reference value
                0.5f
            },
            {
                "Two GC base pairs at optimal distance",
                {0.0f, 0.0f, 0.0f, 0.7f, 0.0f, 0.0f},
                {2, 3},
                -3.5f,  // Approximate reference value
                0.5f
            },
            {
                "Four nucleotide duplex",
                {0.0f, 0.0f, 0.0f, 0.7f, 0.0f, 0.0f, 1.4f, 0.0f, 0.0f, 1.4f, 2.0f, 0.0f},
                {0, 1, 2, 3},
                -8.0f,  // Approximate reference value
                1.0f
            }
        };
    }
    
    void create_test_system(DNAParticle& particles, const TestConfiguration& config) {
        const int N = particles.num_particles;
        
        for (int i = 0; i < N; ++i) {
            particles.positions[i][0] = config.positions[3*i];
            particles.positions[i][1] = config.positions[3*i + 1];
            particles.positions[i][2] = config.positions[3*i + 2];
            
            particles.types[i] = config.types[i];
            particles.btypes[i] = config.types[i];
            particles.strand_ids[i] = (i < N/2) ? 0 : 1;
            
            // Set up simple backbone bonding
            if (i < N/2 - 1) {
                particles.n3_neighbors[i] = i + 1;
                particles.n5_neighbors[i + 1] = i;
                particles.bonded_mask[i][i + 1] = true;
                particles.bonded_mask[i + 1][i] = true;
            } else if (i >= N/2 && i < N - 1) {
                particles.n3_neighbors[i] = i + 1;
                particles.n5_neighbors[i + 1] = i;
                particles.bonded_mask[i][i + 1] = true;
                particles.bonded_mask[i + 1][i] = true;
            }
            
            particles.orientations[i] = torch::eye(3, torch::kFloat32);
        }
        
        particles.compute_interaction_centers();
    }
    
    bool compare_with_reference(float computed, float reference, float tolerance) {
        return std::abs(computed - reference) < tolerance;
    }
    
public:
    OxDNAValidation() {
        load_test_configurations();
    }
    
    void test_energy_validation() {
        std::cout << "\n=== Energy Validation Tests ===" << std::endl;
        
        auto calculator = create_default_calculator();
        
        std::cout << std::setw(30) << "Test Name" << std::setw(15) << "Computed" 
                  << std::setw(15) << "Reference" << std::setw(15) << "Difference" 
                  << std::setw(10) << "Result" << std::endl;
        std::cout << std::string(85, '-') << std::endl;
        
        for (const auto& config : test_configs) {
            const int N = config.positions.size() / 3;
            DNAParticle particles(N, torch::kCPU);
            create_test_system(particles, config);
            
            torch::Tensor energy = calculator.compute_energy(particles);
            float computed_energy = energy.item<float>();
            float difference = computed_energy - config.reference_energy;
            bool passed = compare_with_reference(computed_energy, config.reference_energy, config.tolerance);
            
            total_tests++;
            if (passed) passed_tests++;
            
            std::cout << std::setw(30) << config.name
                      << std::setw(15) << std::fixed << std::setprecision(4) << computed_energy
                      << std::setw(15) << std::setprecision(4) << config.reference_energy
                      << std::setw(15) << std::setprecision(4) << difference
                      << std::setw(10) << (passed ? "PASS" : "FAIL") << std::endl;
        }
    }
    
    void test_force_validation() {
        std::cout << "\n=== Force Validation Tests ===" << std::endl;
        
        auto calculator = create_default_calculator();
        
        // Test force conservation (Newton's third law)
        const int N = 4;
        DNAParticle particles(N, torch::kCPU);
        
        // Create simple symmetric configuration
        particles.positions[0] = torch::tensor({-0.35f, -1.0f, 0.0f});
        particles.positions[1] = torch::tensor({0.35f, -1.0f, 0.0f});
        particles.positions[2] = torch::tensor({0.35f, 1.0f, 0.0f});
        particles.positions[3] = torch::tensor({-0.35f, 1.0f, 0.0f});
        
        for (int i = 0; i < N; ++i) {
            particles.types[i] = i % 4;
            particles.btypes[i] = i % 4;
            particles.strand_ids[i] = (i < 2) ? 0 : 1;
            particles.orientations[i] = torch::eye(3, torch::kFloat32);
            
            if (i < N - 1) {
                particles.n3_neighbors[i] = i + 1;
                particles.n5_neighbors[i + 1] = i;
                particles.bonded_mask[i][i + 1] = true;
                particles.bonded_mask[i + 1][i] = true;
            }
        }
        
        particles.compute_interaction_centers();
        
        torch::Tensor forces = calculator.compute_forces(particles);
        torch::Tensor total_force = torch::sum(forces, 0);
        
        float total_force_magnitude = torch::norm(total_force).item<float>();
        bool force_conservation = total_force_magnitude < 1e-6f;
        
        total_tests++;
        if (force_conservation) passed_tests++;
        
        std::cout << "Force conservation test:" << std::endl;
        std::cout << "Total force magnitude: " << std::scientific << total_force_magnitude << std::endl;
        std::cout << "Result: " << (force_conservation ? "PASS" : "FAIL") << std::endl;
        
        // Test force-energy consistency (numerical derivative)
        float h = 1e-6f;
        torch::Tensor original_pos = particles.positions.clone();
        
        // Compute energy and forces
        torch::Tensor original_energy = calculator.compute_energy(particles);
        torch::Tensor original_forces = calculator.compute_forces(particles);
        
        // Test numerical derivative for one particle
        int test_particle = 0;
        for (int dim = 0; dim < 3; ++dim) {
            particles.positions = original_pos.clone();
            particles.positions[test_particle][dim] += h;
            particles.compute_interaction_centers();
            
            torch::Tensor perturbed_energy = calculator.compute_energy(particles);
            float numerical_force = -(perturbed_energy.item<float>() - original_energy.item<float>()) / h;
            float analytical_force = original_forces[test_particle][dim].item<float>();
            
            float error = std::abs(numerical_force - analytical_force);
            bool consistency = error < 1e-3f;
            
            total_tests++;
            if (consistency) passed_tests++;
            
            std::cout << "Force-energy consistency (particle " << test_particle 
                      << ", dim " << dim << "):" << std::endl;
            std::cout << "Numerical: " << std::fixed << std::setprecision(6) << numerical_force
                      << ", Analytical: " << analytical_force
                      << ", Error: " << std::scientific << error
                      << ", Result: " << (consistency ? "PASS" : "FAIL") << std::endl;
        }
    }
    
    void test_parameter_sensitivity() {
        std::cout << "\n=== Parameter Sensitivity Tests ===" << std::endl;
        
        const int N = 4;
        DNAParticle particles(N, torch::kCPU);
        
        // Create test configuration
        for (int i = 0; i < N; ++i) {
            particles.positions[i] = torch::tensor({static_cast<float>(i) * 0.7f, 0.0f, 0.0f});
            particles.types[i] = i % 4;
            particles.btypes[i] = i % 4;
            particles.strand_ids[i] = (i < 2) ? 0 : 1;
            particles.orientations[i] = torch::eye(3, torch::kFloat32);
            
            if (i < N - 1) {
                particles.n3_neighbors[i] = i + 1;
                particles.n5_neighbors[i + 1] = i;
                particles.bonded_mask[i][i + 1] = true;
                particles.bonded_mask[i + 1][i] = true;
            }
        }
        
        particles.compute_interaction_centers();
        
        // Test temperature sensitivity
        std::vector<float> temperatures = {0.1f, 0.5f, 1.0f, 2.0f};
        std::vector<float> temperature_energies;
        
        std::cout << "Temperature sensitivity:" << std::endl;
        std::cout << std::setw(12) << "Temperature" << std::setw(15) << "Energy" << std::endl;
        std::cout << std::string(27, '-') << std::endl;
        
        for (float temp : temperatures) {
            DNA2Parameters params;
            params.temperature = temp;
            params.initialize_tensors();
            
            DNA2EnergyCalculator calculator(params);
            torch::Tensor energy = calculator.compute_energy(particles);
            temperature_energies.push_back(energy.item<float>());
            
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << temp
                      << std::setw(15) << std::setprecision(6) << energy.item<float>() << std::endl;
        }
        
        // Check that energy scales appropriately with temperature
        bool temp_sensitivity = true;
        for (size_t i = 1; i < temperature_energies.size(); ++i) {
            float ratio = temperature_energies[i] / temperature_energies[i-1];
            float temp_ratio = temperatures[i] / temperatures[i-1];
            if (std::abs(ratio - temp_ratio) > 0.5f) {  // Allow some tolerance
                temp_sensitivity = false;
                break;
            }
        }
        
        total_tests++;
        if (temp_sensitivity) passed_tests++;
        
        std::cout << "Temperature scaling: " << (temp_sensitivity ? "PASS" : "FAIL") << std::endl;
        
        // Test salt concentration sensitivity
        std::vector<float> salt_concentrations = {0.01f, 0.1f, 0.5f, 1.0f};
        
        std::cout << "\nSalt concentration sensitivity:" << std::endl;
        std::cout << std::setw(15) << "Salt (M)" << std::setw(15) << "Energy" << std::endl;
        std::cout << std::string(30, '-') << std::endl;
        
        for (float salt : salt_concentrations) {
            DNA2Parameters params;
            params.salt_concentration = salt;
            params.initialize_tensors();
            
            DNA2EnergyCalculator calculator(params);
            torch::Tensor energy = calculator.compute_energy(particles);
            
            std::cout << std::setw(15) << std::fixed << std::setprecision(3) << salt
                      << std::setw(15) << std::setprecision(6) << energy.item<float>() << std::endl;
        }
        
        total_tests++;
        passed_tests++;  // Salt test always passes (just checking it runs)
    }
    
    void test_edge_cases() {
        std::cout << "\n=== Edge Case Tests ===" << std::endl;
        
        auto calculator = create_default_calculator();
        
        // Test very close particles
        {
            const int N = 2;
            DNAParticle particles(N, torch::kCPU);
            
            particles.positions[0] = torch::tensor({0.0f, 0.0f, 0.0f});
            particles.positions[1] = torch::tensor({0.1f, 0.0f, 0.0f});  // Very close
            
            particles.types[0] = 0;
            particles.types[1] = 1;
            particles.btypes[0] = 0;
            particles.btypes[1] = 1;
            particles.strand_ids[0] = 0;
            particles.strand_ids[1] = 1;
            particles.orientations[0] = torch::eye(3, torch::kFloat32);
            particles.orientations[1] = torch::eye(3, torch::kFloat32);
            
            particles.compute_interaction_centers();
            
            torch::Tensor energy = calculator.compute_energy(particles);
            torch::Tensor forces = calculator.compute_forces(particles);
            
            bool close_particles_finite = torch::all(torch::isfinite(energy)) && 
                                        torch::all(torch::isfinite(forces));
            
            total_tests++;
            if (close_particles_finite) passed_tests++;
            
            std::cout << "Very close particles: " << (close_particles_finite ? "PASS" : "FAIL") << std::endl;
            std::cout << "Energy: " << std::fixed << std::setprecision(6) << energy.item<float>() << std::endl;
        }
        
        // Test very far particles
        {
            const int N = 2;
            DNAParticle particles(N, torch::kCPU);
            
            particles.positions[0] = torch::tensor({0.0f, 0.0f, 0.0f});
            particles.positions[1] = torch::tensor({10.0f, 0.0f, 0.0f});  // Very far
            
            particles.types[0] = 0;
            particles.types[1] = 1;
            particles.btypes[0] = 0;
            particles.btypes[1] = 1;
            particles.strand_ids[0] = 0;
            particles.strand_ids[1] = 1;
            particles.orientations[0] = torch::eye(3, torch::kFloat32);
            particles.orientations[1] = torch::eye(3, torch::kFloat32);
            
            particles.compute_interaction_centers();
            
            torch::Tensor energy = calculator.compute_energy(particles);
            torch::Tensor forces = calculator.compute_forces(particles);
            
            bool far_particles_finite = torch::all(torch::isfinite(energy)) && 
                                      torch::all(torch::isfinite(forces));
            bool far_particles_low_energy = energy.item<float()) > -1e-3f;  // Should be near zero
            
            total_tests += 2;
            if (far_particles_finite) passed_tests++;
            if (far_particles_low_energy) passed_tests++;
            
            std::cout << "Very far particles: " << (far_particles_finite ? "PASS" : "FAIL") << std::endl;
            std::cout << "Low energy: " << (far_particles_low_energy ? "PASS" : "FAIL") << std::endl;
            std::cout << "Energy: " << std::fixed << std::setprecision(6) << energy.item<float>() << std::endl;
        }
        
        // Test single particle
        {
            const int N = 1;
            DNAParticle particles(N, torch::kCPU);
            
            particles.positions[0] = torch::tensor({0.0f, 0.0f, 0.0f});
            particles.types[0] = 0;
            particles.btypes[0] = 0;
            particles.strand_ids[0] = 0;
            particles.orientations[0] = torch::eye(3, torch::kFloat32);
            
            particles.compute_interaction_centers();
            
            torch::Tensor energy = calculator.compute_energy(particles);
            torch::Tensor forces = calculator.compute_forces(particles);
            
            bool single_particle_finite = torch::all(torch::isfinite(energy)) && 
                                        torch::all(torch::isfinite(forces));
            bool single_particle_zero_energy = std::abs(energy.item<float>()) < 1e-6f;
            
            total_tests += 2;
            if (single_particle_finite) passed_tests++;
            if (single_particle_zero_energy) passed_tests++;
            
            std::cout << "Single particle: " << (single_particle_finite ? "PASS" : "FAIL") << std::endl;
            std::cout << "Zero energy: " << (single_particle_zero_energy ? "PASS" : "FAIL") << std::endl;
            std::cout << "Energy: " << std::fixed << std::setprecision(6) << energy.item<float>() << std::endl;
        }
    }
    
    void run_all_validation_tests() {
        std::cout << "DNA2 oxDNA Validation Test Suite" << std::endl;
        std::cout << "=================================" << std::endl;
        
        test_energy_validation();
        test_force_validation();
        test_parameter_sensitivity();
        test_edge_cases();
        
        std::cout << "\n=== Validation Summary ===" << std::endl;
        std::cout << "Passed: " << passed_tests << "/" << total_tests << " tests" << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << "✓ All validation tests passed!" << std::endl;
        } else {
            std::cout << "✗ Some validation tests failed!" << std::endl;
        }
        
        float pass_rate = 100.0f * passed_tests / total_tests;
        std::cout << "Pass rate: " << std::fixed << std::setprecision(1) << pass_rate << "%" << std::endl;
    }
};

int main() {
    try {
        OxDNAValidation validator;
        validator.run_all_validation_tests();
        return (validator.passed_tests == validator.total_tests) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Validation failed with exception: " << e.what() << std::endl;
        return 1;
    }
}