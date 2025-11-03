#pragma once

#include <torch/torch.h>
#include "dna2_data_structures.h"
#include "dna2_mathematical_functions.h"
#include "dna2_mesh_interpolation.h"
#include <memory>
#include <vector>

namespace dna2 {

/**
 * @brief Abstract base class for all DNA2 interactions
 */
class BaseInteraction {
public:
    explicit BaseInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    virtual ~BaseInteraction() = default;
    
    /**
     * @brief Compute energy for this interaction type
     * @param particles Particle data
     * @param distances Pre-computed distance matrix
     * @param params DNA2 parameters
     * @return Energy tensor
     */
    virtual torch::Tensor compute_energy(const DNAParticle& particles,
                                       const torch::Tensor& distances,
                                       const DNA2Parameters& params) = 0;
    
    /**
     * @brief Compute forces and torques for this interaction type
     * @param particles Particle data
     * @param distances Pre-computed distance matrix
     * @param params DNA2 parameters
     * @return Pair of (forces, torques)
     */
    virtual std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) = 0;
    
    /**
     * @brief Check if this interaction is applicable to the given particles
     * @param particles Particle data
     * @return Boolean mask of applicable particle pairs
     */
    virtual torch::Tensor check_applicability(const DNAParticle& particles) = 0;
    
    /**
     * @brief Get interaction type
     */
    virtual InteractionType get_type() const = 0;

protected:
    std::shared_ptr<MeshInterpolator> mesh_interpolator_;
    
    // Helper functions
    torch::Tensor get_bonded_pairs(const DNAParticle& particles);
    torch::Tensor get_non_bonded_pairs(const DNAParticle& particles, float cutoff);
    torch::Tensor get_candidate_pairs(const DNAParticle& particles, float cutoff);
};

/**
 * @brief Backbone (FENE) interaction
 */
class BackboneInteraction : public BaseInteraction {
public:
    explicit BackboneInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    torch::Tensor compute_energy(const DNAParticle& particles,
                               const torch::Tensor& distances,
                               const DNA2Parameters& params) override;
    
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) override;
    
    torch::Tensor check_applicability(const DNAParticle& particles) override;
    InteractionType get_type() const override { return InteractionType::BACKBONE; }

private:
    torch::Tensor compute_fene_energy(const torch::Tensor& r_backbone,
                                    const DNA2Parameters& params);
    
    std::pair<torch::Tensor, torch::Tensor> compute_fene_forces(
        const torch::Tensor& r_backbone,
        const torch::Tensor& r_backbone_norm,
        const DNA2Parameters& params);
};

/**
 * @brief Stacking interaction
 */
class StackingInteraction : public BaseInteraction {
public:
    explicit StackingInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    torch::Tensor compute_energy(const DNAParticle& particles,
                               const torch::Tensor& distances,
                               const DNA2Parameters& params) override;
    
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) override;
    
    torch::Tensor check_applicability(const DNAParticle& particles) override;
    InteractionType get_type() const override { return InteractionType::STACKING; }

private:
    torch::Tensor compute_angular_contributions(const DNAParticle& particles,
                                              const torch::Tensor& pair_indices,
                                              const DNA2Parameters& params);
    
    std::pair<torch::Tensor, torch::Tensor> compute_stacking_forces(
        const DNAParticle& particles,
        const torch::Tensor& pair_indices,
        const torch::Tensor& distances,
        const DNA2Parameters& params);
};

/**
 * @brief Hydrogen bonding interaction
 */
class HydrogenBondingInteraction : public BaseInteraction {
public:
    explicit HydrogenBondingInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    torch::Tensor compute_energy(const DNAParticle& particles,
                               const torch::Tensor& distances,
                               const DNA2Parameters& params) override;
    
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) override;
    
    torch::Tensor check_applicability(const DNAParticle& particles) override;
    InteractionType get_type() const override { return InteractionType::HYDROGEN_BONDING; }

private:
    torch::Tensor check_base_pairing(const DNAParticle& particles,
                                   const torch::Tensor& pair_indices);
    
    torch::Tensor compute_hb_angular_terms(const DNAParticle& particles,
                                         const torch::Tensor& pair_indices,
                                         const DNA2Parameters& params);
};

/**
 * @brief Excluded volume interaction
 */
class ExcludedVolumeInteraction : public BaseInteraction {
public:
    explicit ExcludedVolumeInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    torch::Tensor compute_energy(const DNAParticle& particles,
                               const torch::Tensor& distances,
                               const DNA2Parameters& params) override;
    
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) override;
    
    torch::Tensor check_applicability(const DNAParticle& particles) override;
    InteractionType get_type() const override { return InteractionType::EXCLUDED_VOLUME; }

private:
    torch::Tensor compute_lj_energy(const torch::Tensor& r,
                                  const torch::Tensor& sigma,
                                  const torch::Tensor& epsilon,
                                  const torch::Tensor& rstar,
                                  const torch::Tensor& b,
                                  const torch::Tensor& rc);
    
    torch::Tensor compute_lj_force(const torch::Tensor& r,
                                 const torch::Tensor& sigma,
                                 const torch::Tensor& epsilon,
                                 const torch::Tensor& rstar,
                                 const torch::Tensor& b,
                                 const torch::Tensor& rc);
};

/**
 * @brief Cross-stacking interaction
 */
class CrossStackingInteraction : public BaseInteraction {
public:
    explicit CrossStackingInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    torch::Tensor compute_energy(const DNAParticle& particles,
                               const torch::Tensor& distances,
                               const DNA2Parameters& params) override;
    
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) override;
    
    torch::Tensor check_applicability(const DNAParticle& particles) override;
    InteractionType get_type() const override { return InteractionType::CROSS_STACKING; }

private:
    torch::Tensor compute_crst_angular_terms(const DNAParticle& particles,
                                           const torch::Tensor& pair_indices,
                                           const DNA2Parameters& params);
};

/**
 * @brief Coaxial stacking interaction
 */
class CoaxialStackingInteraction : public BaseInteraction {
public:
    explicit CoaxialStackingInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    torch::Tensor compute_energy(const DNAParticle& particles,
                               const torch::Tensor& distances,
                               const DNA2Parameters& params) override;
    
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) override;
    
    torch::Tensor check_applicability(const DNAParticle& particles) override;
    InteractionType get_type() const override { return InteractionType::COAXIAL_STACKING; }

private:
    torch::Tensor compute_pure_harmonic(const torch::Tensor& theta,
                                      FunctionType type,
                                      const DNA2Parameters& params);
};

/**
 * @brief Debye-Huckel electrostatic interaction
 */
class DebyeHuckelInteraction : public BaseInteraction {
public:
    explicit DebyeHuckelInteraction(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    torch::Tensor compute_energy(const DNAParticle& particles,
                               const torch::Tensor& distances,
                               const DNA2Parameters& params) override;
    
    std::pair<torch::Tensor, torch::Tensor> compute_forces_and_torques(
        const DNAParticle& particles,
        const torch::Tensor& distances,
        const DNA2Parameters& params) override;
    
    torch::Tensor check_applicability(const DNAParticle& particles) override;
    InteractionType get_type() const override { return InteractionType::DEBYE_HUCKEL; }

private:
    float compute_debye_length(const DNA2Parameters& params);
    torch::Tensor compute_dh_energy(const torch::Tensor& r,
                                  float debye_length,
                                  float strength,
                                  const torch::Tensor& charges);
    
    torch::Tensor compute_dh_force(const torch::Tensor& r,
                                 float debye_length,
                                 float strength,
                                 const torch::Tensor& charges);
};

/**
 * @brief Interaction manager for coordinating all interaction types
 */
class InteractionManager {
public:
    explicit InteractionManager(std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    /**
     * @brief Compute total energy from all interactions
     */
    InteractionResult compute_total_energy(const DNAParticle& particles,
                                         const DNA2Parameters& params,
                                         bool compute_forces = true,
                                         bool compute_torques = true);
    
    /**
     * @brief Compute specific interaction term
     */
    torch::Tensor compute_interaction_term(InteractionType type,
                                         const DNAParticle& particles,
                                         const DNA2Parameters& params);
    
    /**
     * @brief Enable/disable specific interactions
     */
    void enable_interaction(InteractionType type, bool enable = true);
    void disable_interaction(InteractionType type) { enable_interaction(type, false); }
    
    /**
     * @brief Get interaction instance
     */
    std::shared_ptr<BaseInteraction> get_interaction(InteractionType type);

private:
    std::unordered_map<InteractionType, std::shared_ptr<BaseInteraction>> interactions_;
    std::unordered_map<InteractionType, bool> interaction_enabled_;
    
    torch::Tensor compute_pairwise_distances(const DNAParticle& particles);
    void initialize_interactions(std::shared_ptr<MeshInterpolator> mesh_interpolator);
};

/**
 * @brief Factory for creating interaction instances
 */
class InteractionFactory {
public:
    static std::unique_ptr<BaseInteraction> create_interaction(
        InteractionType type,
        std::shared_ptr<MeshInterpolator> mesh_interpolator);
    
    static std::unique_ptr<InteractionManager> create_manager(
        std::shared_ptr<MeshInterpolator> mesh_interpolator);
};

} // namespace dna2