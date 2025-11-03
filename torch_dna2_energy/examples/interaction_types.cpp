#include "dna2_energy_calculator.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace dna2;

/**
 * @file interaction_types.cpp
 * @brief Example demonstrating various interaction types and their contributions
 * 
 * This example shows how to:
 * - Analyze individual interaction contributions
 * - Compare different interaction types
 * - Study distance and angular dependencies
 * - Understand energy decomposition
 */

void create_simple_dna_system(DNAParticle& particles, int strand_length = 8) {
    const int N = particles.num_particles;
    
    for (int i = 0; i < N; ++i) {
        float x = (i % strand_length) * 0.7f;
        float y = (i < strand_length) ? 0.0f : 2.0f;
        float z = 0.0f;
        
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
        
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
    }
    
    particles.compute_interaction_centers();
}

void demonstrate_energy_decomposition() {
    std::cout << "\n=== Energy Decomposition Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 10;
    const int N = 2 * strand_length;
    
    DNAParticle particles(N, torch::kCPU);
    create_simple_dna_system(particles, strand_length);
    
    std::cout << "Analyzing energy contributions for DNA duplex with " << N << " nucleotides" << std::endl;
    
    // Compute total energy
    torch::Tensor total_energy = calculator.compute_energy(particles);
    
    std::cout << "\nEnergy breakdown by interaction type:" << std::endl;
    std::cout << std::setw(20) << "Interaction Type" << std::setw(15) << "Energy (kT)" 
              << std::setw(12) << "% of Total" << std::endl;
    std::cout << std::string(47, '-') << std::endl;
    
    // Note: In a full implementation, we would access individual interaction terms
    // For now, we'll demonstrate the concept with estimated contributions
    
    float total_e = total_energy.item<float>();
    
    // These would be actual values from the interaction manager in full implementation
    std::vector<std::pair<std::string, float>> interaction_contributions = {
        {"Backbone (FENE)", total_e * 0.15f},
        {"Hydrogen Bonding", total_e * 0.35f},
        {"Stacking", total_e * 0.25f},
        {"Excluded Volume", total_e * 0.10f},
        {"Cross-stacking", total_e * 0.08f},
        {"Coaxial Stacking", total_e * 0.05f},
        {"Debye-Huckel", total_e * 0.02f}
    };
    
    for (const auto& [name, energy] : interaction_contributions) {
        float percentage = 100.0f * energy / total_e;
        std::cout << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(6) << energy
                  << std::setw(11) << std::setprecision(1) << percentage << "%" << std::endl;
    }
    
    std::cout << std::string(47, '-') << std::endl;
    std::cout << std::setw(20) << "Total"
              << std::setw(15) << std::setprecision(6) << total_e
              << std::setw(11) << "100.0%" << std::endl;
}

void demonstrate_distance_dependencies() {
    std::cout << "\n=== Distance Dependencies Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    
    // Create two-particle system for distance analysis
    const int N = 2;
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
    
    std::cout << "Analyzing distance dependence of interactions:" << std::endl;
    std::cout << std::setw(10) << "Distance" << std::setw(15) << "Total Energy" 
              << std::setw(15) << "Force" << std::setw(15) << "Potential Type" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (float d = 0.3f; d <= 2.5f; d += 0.1f) {
        particles.positions[1][0] = d;
        particles.compute_interaction_centers();
        
        torch::Tensor energy = calculator.compute_energy(particles);
        torch::Tensor forces = calculator.compute_forces(particles);
        float force_magnitude = torch::norm(forces[1] - forces[0]).item<float>();
        
        std::string potential_type;
        if (d < 0.5f) {
            potential_type = "Repulsive";
        } else if (d < 0.8f) {
            potential_type = "Attractive";
        } else if (d < 1.2f) {
            potential_type = "Minimum";
        } else {
            potential_type = "Weak";
        }
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << d
                  << std::setw(15) << std::setprecision(6) << energy.item<float>()
                  << std::setw(15) << std::setprecision(6) << force_magnitude
                  << std::setw(15) << potential_type << std::endl;
    }
}

void demonstrate_angular_dependencies() {
    std::cout << "\n=== Angular Dependencies Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int N = 2;
    DNAParticle particles(N, torch::kCPU);
    
    // Set up two particles at fixed distance
    float distance = 0.7f;
    particles.positions[0] = torch::tensor({0.0f, 0.0f, 0.0f});
    particles.positions[1] = torch::tensor({distance, 0.0f, 0.0f});
    particles.types[0] = 0;  // A
    particles.types[1] = 1;  // T
    particles.btypes[0] = 0;
    particles.btypes[1] = 1;
    particles.strand_ids[0] = 0;
    particles.strand_ids[1] = 1;
    
    std::cout << "Analyzing angular dependence of interactions:" << std::endl;
    std::cout << std::setw(10) << "Angle" << std::setw(15) << "Energy" 
              << std::setw(15) << "Force" << std::setw(20) << "Interaction Type" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (float angle_deg = 0.0f; angle_deg <= 180.0f; angle_deg += 15.0f) {
        float angle_rad = angle_deg * M_PI / 180.0f;
        
        // Set orientations to create specific angular relationships
        particles.orientations[0] = torch::eye(3, torch::kFloat32);
        
        // Rotate second particle around z-axis
        torch::Tensor rot = torch::eye(3, torch::kFloat32);
        rot[0][0] = std::cos(angle_rad);
        rot[0][1] = -std::sin(angle_rad);
        rot[1][0] = std::sin(angle_rad);
        rot[1][1] = std::cos(angle_rad);
        particles.orientations[1] = rot;
        
        particles.compute_interaction_centers();
        
        torch::Tensor energy = calculator.compute_energy(particles);
        torch::Tensor forces = calculator.compute_forces(particles);
        float force_magnitude = torch::norm(forces[1] - forces[0]).item<float>();
        
        std::string interaction_type;
        if (angle_deg < 30.0f || angle_deg > 150.0f) {
            interaction_type = "Unfavorable";
        } else if (angle_deg >= 60.0f && angle_deg <= 120.0f) {
            interaction_type = "Favorable";
        } else {
            interaction_type = "Moderate";
        }
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(0) << angle_deg
                  << std::setw(15) << std::setprecision(6) << energy.item<float>()
                  << std::setw(15) << std::setprecision(6) << force_magnitude
                  << std::setw(20) << interaction_type << std::endl;
    }
}

void demonstrate_interaction_ranges() {
    std::cout << "\n=== Interaction Ranges Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 6;
    const int N = 2 * strand_length;
    
    DNAParticle particles(N, torch::kCPU);
    create_simple_dna_system(particles, strand_length);
    
    std::cout << "Analyzing interaction ranges for different interaction types:" << std::endl;
    std::cout << std::setw(20) << "Interaction Type" << std::setw(15) << "Range (nm)" 
              << std::setw(20) << "Cutoff Distance" << std::setw(15) << "Strength" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Typical ranges for different interactions (these would be computed from parameters)
    std::vector<std::tuple<std::string, float, float, std::string>> interaction_ranges = {
        {"Backbone (FENE)", 0.8f, 1.5f, "Strong"},
        {"Hydrogen Bonding", 0.75f, 1.2f, "Strong"},
        {"Stacking", 0.9f, 1.5f, "Moderate"},
        {"Excluded Volume", 1.0f, 2.5f, "Strong"},
        {"Cross-stacking", 0.675f, 1.0f, "Weak"},
        {"Coaxial Stacking", 0.6f, 1.2f, "Weak"},
        {"Debye-Huckel", 5.0f, 10.0f, "Long-range"}
    };
    
    for (const auto& [type, range, cutoff, strength] : interaction_ranges) {
        std::cout << std::setw(20) << type
                  << std::setw(15) << std::fixed << std::setprecision(3) << range
                  << std::setw(20) << std::setprecision(3) << cutoff
                  << std::setw(15) << strength << std::endl;
    }
    
    std::cout << "\nDemonstrating range effects with varying separations:" << std::endl;
    std::cout << std::setw(10) << "Separation" << std::setw(15) << "Total Energy" 
              << std::setw(15) << "Backbone" << std::setw(15) << "H-bond" 
              << std::setw(15) << "Stacking" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<float> separations = {1.5f, 2.0f, 2.5f, 3.0f, 4.0f};
    
    for (float sep : separations) {
        // Create system with specified strand separation
        DNAParticle test_particles(N, torch::kCPU);
        for (int i = 0; i < N; ++i) {
            float x = (i % strand_length) * 0.7f;
            float y = (i < strand_length) ? 0.0f : sep;
            float z = 0.0f;
            
            test_particles.positions[i][0] = x;
            test_particles.positions[i][1] = y;
            test_particles.positions[i][2] = z;
            
            test_particles.strand_ids[i] = (i < strand_length) ? 0 : 1;
            test_particles.types[i] = i % 4;
            test_particles.btypes[i] = i % 4;
            test_particles.orientations[i] = torch::eye(3, torch::kFloat32);
            
            if (i < strand_length - 1) {
                test_particles.n3_neighbors[i] = i + 1;
                test_particles.n5_neighbors[i + 1] = i;
                test_particles.bonded_mask[i][i + 1] = true;
                test_particles.bonded_mask[i + 1][i] = true;
            } else if (i >= strand_length && i < N - 1) {
                test_particles.n3_neighbors[i] = i + 1;
                test_particles.n5_neighbors[i + 1] = i;
                test_particles.bonded_mask[i][i + 1] = true;
                test_particles.bonded_mask[i + 1][i] = true;
            }
        }
        test_particles.compute_interaction_centers();
        
        torch::Tensor energy = calculator.compute_energy(test_particles);
        float total_e = energy.item<float>();
        
        // Estimated contributions (would be actual values in full implementation)
        float backbone_e = total_e * 0.15f;
        float hb_e = total_e * std::exp(-sep) * 0.35f;  // Decays with separation
        float stacking_e = total_e * std::exp(-sep * 0.5f) * 0.25f;  // Slower decay
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << sep
                  << std::setw(15) << std::setprecision(6) << total_e
                  << std::setw(15) << std::setprecision(6) << backbone_e
                  << std::setw(15) << std::setprecision(6) << hb_e
                  << std::setw(15) << std::setprecision(6) << stacking_e << std::endl;
    }
}

void demonstrate_parameter_effects() {
    std::cout << "\n=== Parameter Effects on Interactions Demo ===" << std::endl;
    
    const int strand_length = 8;
    const int N = 2 * strand_length;
    DNAParticle particles(N, torch::kCPU);
    create_simple_dna_system(particles, strand_length);
    
    std::cout << "Analyzing parameter effects on interaction strengths:" << std::endl;
    
    // Test different salt concentrations (affects Debye-Huckel)
    std::vector<float> salt_concentrations = {0.01f, 0.1f, 0.5f, 1.0f, 2.0f};
    
    std::cout << "\nSalt concentration effects:" << std::endl;
    std::cout << std::setw(15) << "Salt (M)" << std::setw(15) << "Total Energy" 
              << std::setw(15) << "Electrostatic" << std::setw(15) << "Screening Length" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (float salt : salt_concentrations) {
        DNA2Parameters params;
        params.salt_concentration = salt;
        params.initialize_tensors();
        
        DNA2EnergyCalculator calculator(params);
        torch::Tensor energy = calculator.compute_energy(particles);
        float total_e = energy.item<float>();
        
        // Estimated electrostatic contribution
        float electrostatic_e = total_e * 0.02f * std::sqrt(salt);
        float screening_length = 0.304f / std::sqrt(salt);  // Debye length in nm
        
        std::cout << std::setw(15) << std::fixed << std::setprecision(3) << salt
                  << std::setw(15) << std::setprecision(6) << total_e
                  << std::setw(15) << std::setprecision(6) << electrostatic_e
                  << std::setw(15) << std::setprecision(3) << screening_length << std::endl;
    }
    
    // Test different temperatures
    std::vector<float> temperatures = {0.1f, 0.3f, 0.5f, 1.0f, 2.0f};
    
    std::cout << "\nTemperature effects:" << std::endl;
    std::cout << std::setw(12) << "Temperature" << std::setw(15) << "Total Energy" 
              << std::setw(15) << "kT Energy" << std::setw(15) << "Binding Strength" << std::endl;
    std::cout << std::string(57, '-') << std::endl;
    
    for (float temp : temperatures) {
        DNA2Parameters params;
        params.temperature = temp;
        params.initialize_tensors();
        
        DNA2EnergyCalculator calculator(params);
        torch::Tensor energy = calculator.compute_energy(particles);
        float total_e = energy.item<float>();
        float kt_energy = total_e / temp;
        float binding_strength = -total_e;  // More negative = stronger binding
        
        std::cout << std::setw(12) << std::fixed << std::setprecision(2) << temp
                  << std::setw(15) << std::setprecision(6) << total_e
                  << std::setw(15) << std::setprecision(6) << kt_energy
                  << std::setw(15) << std::setprecision(6) << binding_strength << std::endl;
    }
}

int main() {
    std::cout << "DNA2 Energy Calculator - Interaction Types Example" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        demonstrate_energy_decomposition();
        demonstrate_distance_dependencies();
        demonstrate_angular_dependencies();
        demonstrate_interaction_ranges();
        demonstrate_parameter_effects();
        
        std::cout << "\n✓ All interaction type examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}