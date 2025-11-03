#include "dna2_data_structures.h"
#include <stdexcept>
#include <cmath>

namespace dna2 {

// DNA2Parameters implementation
DNA2Parameters::DNA2Parameters() {
    initialize_tensors();
}

void DNA2Parameters::initialize_tensors() {
    // Initialize excluded volume parameters
    excl_sigma = torch::tensor({0.574f, 0.574f, 0.574f, 0.574f}, torch::TensorOptions().device(device));
    excl_rstar = torch::tensor({0.68f, 0.68f, 0.68f, 0.68f}, torch::TensorOptions().device(device));
    excl_b = torch::tensor({1.6f, 1.6f, 1.6f, 1.6f}, torch::TensorOptions().device(device));
    excl_rc = torch::tensor({2.0f, 2.0f, 2.0f, 2.0f}, torch::TensorOptions().device(device));
    
    // Initialize hydrogen bonding epsilon matrix (5x5 for sequence dependence)
    hydr_eps_matrix = torch::full({5, 5}, hydr_eps, torch::TensorOptions().device(device));
    
    // Initialize stacking epsilon matrix (5x5 for sequence dependence)
    stck_eps_matrix = torch::full({5, 5}, stck_base_eps, torch::TensorOptions().device(device));
}

void DNA2Parameters::to(torch::Device new_device) {
    device = new_device;
    
    excl_sigma = excl_sigma.to(new_device);
    excl_rstar = excl_rstar.to(new_device);
    excl_b = excl_b.to(new_device);
    excl_rc = excl_rc.to(new_device);
    hydr_eps_matrix = hydr_eps_matrix.to(new_device);
    stck_eps_matrix = stck_eps_matrix.to(new_device);
}

// DNAParticle implementation
DNAParticle::DNAParticle(int64_t N, torch::Device device) : num_particles(N) {
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    
    // Initialize position and orientation tensors
    positions = torch::zeros({N, 3}, options);
    orientations = torch::zeros({N, 3, 3}, options);
    orientations_t = torch::zeros({N, 3, 3}, options);
    
    // Initialize interaction centers
    backbone_centers = torch::zeros({N, 3}, options);
    stack_centers = torch::zeros({N, 3}, options);
    base_centers = torch::zeros({N, 3}, options);
    
    // Initialize particle properties
    types = torch::zeros({N}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    btypes = torch::zeros({N}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    strand_ids = torch::zeros({N}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    
    // Initialize bonding information
    n3_neighbors = torch::full({N}, -1, torch::TensorOptions().device(device).dtype(torch::kInt64));
    n5_neighbors = torch::full({N}, -1, torch::TensorOptions().device(device).dtype(torch::kInt64));
    bonded_mask = torch::zeros({N, N}, torch::TensorOptions().device(device).dtype(torch::kBool));
}

void DNAParticle::compute_interaction_centers() {
    // Compute interaction centers from positions and orientations
    // This is a simplified implementation - in practice, this would use
    // the specific geometry of DNA nucleotides
    
    // For now, assume standard DNA geometry
    // Backbone center: position + orientation * (0, 0, 0.4)
    // Stack center: position + orientation * (0, 0, 0.0)  
    // Base center: position + orientation * (0, 0, -0.4)
    
    auto z_axis = orientations.select(2, 2);  // Third column of rotation matrix
    auto backbone_offset = z_axis * 0.4f;
    auto base_offset = z_axis * (-0.4f);
    
    backbone_centers = positions + backbone_offset;
    stack_centers = positions;  // Stacking at particle center
    base_centers = positions + base_offset;
    
    // Update orientations transpose
    orientations_t = orientations.transpose(-2, -1);
}

void DNAParticle::to(torch::Device new_device) {
    positions = positions.to(new_device);
    orientations = orientations.to(new_device);
    orientations_t = orientations_t.to(new_device);
    backbone_centers = backbone_centers.to(new_device);
    stack_centers = stack_centers.to(new_device);
    base_centers = base_centers.to(new_device);
    types = types.to(new_device);
    btypes = btypes.to(new_device);
    strand_ids = strand_ids.to(new_device);
    n3_neighbors = n3_neighbors.to(new_device);
    n5_neighbors = n5_neighbors.to(new_device);
    bonded_mask = bonded_mask.to(new_device);
}

void DNAParticle::validate() const {
    // Check tensor shapes
    if (positions.sizes() != std::vector<int64_t>{num_particles, 3}) {
        throw DNA2EnergyException("Invalid positions tensor shape");
    }
    if (orientations.sizes() != std::vector<int64_t>{num_particles, 3, 3}) {
        throw DNA2EnergyException("Invalid orientations tensor shape");
    }
    if (types.sizes() != std::vector<int64_t>{num_particles}) {
        throw DNA2EnergyException("Invalid types tensor shape");
    }
    
    // Check device consistency
    auto device = positions.device();
    if (orientations.device() != device || backbone_centers.device() != device ||
        stack_centers.device() != device || base_centers.device() != device) {
        throw DNA2EnergyException("Inconsistent tensor devices");
    }
    
    // Check value ranges
    auto max_type = torch::max(types).item<int>();
    auto min_type = torch::min(types).item<int>();
    if (min_type < 0 || max_type > 4) {
        throw DNA2EnergyException("Particle types must be in range [0, 4]");
    }
}

// InteractionResult implementation
InteractionResult::InteractionResult(int64_t N, int64_t batch_size, torch::Device device) {
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    
    total_energy = torch::zeros({batch_size}, options);
    forces = torch::zeros({batch_size, N, 3}, options);
    torques = torch::zeros({batch_size, N, 3}, options);
    
    // Energy breakdown tensors
    backbone_energy = torch::zeros({batch_size}, options);
    stacking_energy = torch::zeros({batch_size}, options);
    hb_energy = torch::zeros({batch_size}, options);
    excl_energy = torch::zeros({batch_size}, options);
    crst_energy = torch::zeros({batch_size}, options);
    cxst_energy = torch::zeros({batch_size}, options);
    dh_energy = torch::zeros({batch_size}, options);
}

void InteractionResult::to(torch::Device new_device) {
    total_energy = total_energy.to(new_device);
    forces = forces.to(new_device);
    torques = torques.to(new_device);
    backbone_energy = backbone_energy.to(new_device);
    stacking_energy = stacking_energy.to(new_device);
    hb_energy = hb_energy.to(new_device);
    excl_energy = excl_energy.to(new_device);
    crst_energy = crst_energy.to(new_device);
    cxst_energy = cxst_energy.to(new_device);
    dh_energy = dh_energy.to(new_device);
}

void InteractionResult::reset() {
    total_energy.zero_();
    forces.zero_();
    torques.zero_();
    backbone_energy.zero_();
    stacking_energy.zero_();
    hb_energy.zero_();
    excl_energy.zero_();
    crst_energy.zero_();
    cxst_energy.zero_();
    dh_energy.zero_();
}

} // namespace dna2