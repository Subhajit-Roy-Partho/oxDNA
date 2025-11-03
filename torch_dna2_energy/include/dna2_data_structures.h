#pragma once

#include <torch/torch.h>
#include <vector>

namespace dna2 {

/**
 * @brief DNA2 model parameters structure
 * 
 * Contains all parameters needed for DNA2 energy calculations
 */
struct DNA2Parameters {
    // FENE parameters for backbone bonds
    float fene_eps = 2.0f;
    float fene_r0 = 0.7564f;  // oxDNA2 specific
    float fene_delta = 0.25f;
    
    // Excluded volume parameters
    float excl_eps = 2.0f;
    torch::Tensor excl_sigma;    // [4] - sigma for different site types
    torch::Tensor excl_rstar;    // [4] - rstar for different site types
    torch::Tensor excl_b;        // [4] - b parameter for different site types
    torch::Tensor excl_rc;       // [4] - cutoff for different site types
    
    // Hydrogen bonding parameters
    float hydr_eps = 1.0678f;    // oxDNA2 specific
    float hydr_a = 8.0f;
    float hydr_rc = 0.75f;
    float hydr_r0 = 0.4f;
    torch::Tensor hydr_eps_matrix; // [5, 5] - sequence-dependent epsilons
    
    // Stacking parameters
    float stck_base_eps = 1.3523f;  // oxDNA2 specific
    float stck_fact_eps = 2.6717f;  // oxDNA2 specific
    float stck_a = 6.0f;
    float stck_rc = 0.9f;
    float stck_r0 = 0.4f;
    torch::Tensor stck_eps_matrix; // [5, 5] - sequence-dependent epsilons
    
    // Cross-stacking parameters
    float crst_k = 47.5f;
    float crst_r0 = 0.575f;
    float crst_rc = 0.675f;
    
    // Coaxial stacking parameters
    float cxst_k = 58.5f;  // oxDNA2 specific
    float cxst_r0 = 0.4f;
    float cxst_rc = 0.6f;
    
    // Debye-Huckel parameters
    float salt_concentration = 0.5f;
    float dh_lambda = 0.3616455f;
    float dh_strength = 0.0543f;
    bool dh_half_charged_ends = true;
    
    // Temperature
    float temperature = 0.1f;  // oxDNA reduced units
    
    // Grooving
    bool grooving = true;
    
    // Device for tensor storage
    torch::Device device = torch::kCPU;
    
    // Constructor
    DNA2Parameters();
    
    // Initialize tensors with proper device
    void initialize_tensors();
    
    // Move all tensors to specified device
    void to(torch::Device device);
};

/**
 * @brief DNA particle data structure
 * 
 * Contains all particle information needed for energy calculations
 */
struct DNAParticle {
    // Position and orientation tensors
    torch::Tensor positions;        // [N, 3] - particle positions
    torch::Tensor orientations;     // [N, 3, 3] - rotation matrices
    torch::Tensor orientations_t;   // [N, 3, 3] - transpose of orientations
    
    // Interaction centers
    torch::Tensor backbone_centers; // [N, 3] - backbone positions
    torch::Tensor stack_centers;    // [N, 3] - stacking positions
    torch::Tensor base_centers;     // [N, 3] - base positions
    
    // Particle properties
    torch::Tensor types;            // [N] - particle types (0-4)
    torch::Tensor btypes;           // [N] - base types for pairing
    torch::Tensor strand_ids;       // [N] - strand identifiers
    
    // Bonding information
    torch::Tensor n3_neighbors;     // [N] - 3' neighbor indices
    torch::Tensor n5_neighbors;     // [N] - 5' neighbor indices
    torch::Tensor bonded_mask;      // [N, N] - bonding matrix
    
    // Number of particles
    int64_t num_particles;
    
    // Constructor
    DNAParticle(int64_t N, torch::Device device = torch::kCPU);
    
    // Compute interaction centers from positions and orientations
    void compute_interaction_centers();
    
    // Move all tensors to specified device
    void to(torch::Device device);
    
    // Validate tensor shapes and devices
    void validate() const;
};

/**
 * @brief Interaction result structure
 * 
 * Contains energy breakdown and force/torque information
 */
struct InteractionResult {
    torch::Tensor total_energy;    // [1] or [batch_size]
    torch::Tensor forces;          // [N, 3] or [batch_size, N, 3]
    torch::Tensor torques;         // [N, 3] or [batch_size, N, 3]
    
    // Energy breakdown by interaction type
    torch::Tensor backbone_energy;
    torch::Tensor stacking_energy;
    torch::Tensor hb_energy;
    torch::Tensor excl_energy;
    torch::Tensor crst_energy;
    torch::Tensor cxst_energy;
    torch::Tensor dh_energy;
    
    // Constructor
    InteractionResult(int64_t N, int64_t batch_size = 1, torch::Device device = torch::kCPU);
    
    // Move all tensors to specified device
    void to(torch::Device device);
    
    // Reset all values to zero
    void reset();
};

// Interaction type constants
enum class InteractionType {
    BACKBONE = 0,
    STACKING = 1,
    HYDROGEN_BONDING = 2,
    EXCLUDED_VOLUME = 3,
    CROSS_STACKING = 4,
    COAXIAL_STACKING = 5,
    DEBYE_HUCKEL = 6
};

// Mesh type constants for angular potentials
enum class MeshType {
    HYDR_F4_THETA1 = 0,
    HYDR_F4_THETA2 = 1,
    HYDR_F4_THETA3 = 2,
    HYDR_F4_THETA4 = 3,
    HYDR_F4_THETA7 = 4,
    HYDR_F4_THETA8 = 5,
    STCK_F4_THETA1 = 6,
    STCK_F4_THETA2 = 7,
    STCK_F4_THETA3 = 8,
    STCK_F4_THETA4 = 9,
    CRST_F4_THETA1 = 10,
    CRST_F4_THETA2 = 11,
    CXST_F4_THETA1 = 12
};

// Function type constants for mathematical functions
enum class FunctionType {
    F1_HYDR = 0,
    F1_STCK = 1,
    F2_CRST = 2,
    F2_CXST = 3,
    F4_HYDR_T1 = 4,
    F4_HYDR_T2 = 5,
    F4_HYDR_T3 = 6,
    F4_HYDR_T4 = 7,
    F4_HYDR_T7 = 8,
    F4_HYDR_T8 = 9,
    F4_STCK_T1 = 10,
    F4_STCK_T2 = 11,
    F4_STCK_T3 = 12,
    F4_STCK_T4 = 13,
    F4_CRST_T1 = 14,
    F4_CRST_T2 = 15,
    F4_CXST_T1 = 16,
    F5_HYDR_PHI = 17,
    F5_STCK_PHI = 18
};

} // namespace dna2