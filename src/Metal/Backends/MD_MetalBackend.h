/**
 * @file    MD_MetalBackend.h
 * @brief   Molecular Dynamics backend for Metal GPU
 *
 * Metal equivalent of MD_CUDABackend
 */

#ifndef MD_METALBACKEND_H_
#define MD_METALBACKEND_H_

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include "MetalBaseBackend.h"
#include "../../Backends/MDBackend.h"
#include "../MetalUtils.h"

/**
 * @brief Manages a MD simulation on Apple GPU with Metal
 *
 * This class implements molecular dynamics simulation using the Metal API
 * for Apple Silicon (M-series) GPUs.
 */
class MD_MetalBackend : public MDBackend, public MetalBaseBackend {
protected:
    bool _use_edge;
    bool _any_rigid_body;
    bool _avoid_cpu_calculations;

    /// Compute pipelines for different kernels
    id<MTLComputePipelineState> _first_step_pipeline;
    id<MTLComputePipelineState> _second_step_pipeline;
    id<MTLComputePipelineState> _forces_pipeline;
    id<MTLComputePipelineState> _zero_forces_pipeline;

    /// Particle velocity and angular momentum buffers
    id<MTLBuffer> _d_vels;      // Linear velocities
    id<MTLBuffer> _d_Ls;        // Angular momenta
    id<MTLBuffer> _d_forces;    // Forces
    id<MTLBuffer> _d_torques;   // Torques

    m_number4 *_h_vels;
    m_number4 *_h_Ls;
    m_number4 *_h_forces;
    m_number4 *_h_torques;

    /// Molecular information for rigid bodies
    std::vector<int> _h_particles_to_mols;
    id<MTLBuffer> _d_particles_to_mols;
    id<MTLBuffer> _d_mol_sizes;
    id<MTLBuffer> _d_molecular_coms;

    /// Sorting buffers
    id<MTLBuffer> _d_buff_vels;
    id<MTLBuffer> _d_buff_Ls;
    id<MTLBuffer> _d_buff_particles_to_mols;

    /// Barostat statistics
    llint _barostat_attempts;
    llint _barostat_accepted;
    int _update_st_every;

    /// Energy output
    bool _print_energy;

    /// Error configuration output
    ObservableOutput *_obs_output_error_conf;
    std::string _error_conf_file;

    /// External forces
    id<MTLBuffer> _d_ext_forces;
    int _max_ext_forces;

    /// Internal methods
    virtual void _gpu_to_host() override;
    virtual void _host_to_gpu() override;
    virtual void _apply_external_forces_changes();

    virtual void _sort_particles();
    virtual void _rescale_molecular_positions(m_number4 new_Ls, m_number4 old_Ls, bool is_reverse_move);
    virtual void _rescale_positions(m_number4 new_Ls, m_number4 old_Ls);

    virtual void _first_step();
    virtual void _apply_barostat();
    virtual void _forces_second_step();
    virtual void _set_external_forces();

    virtual void _thermalize();
    virtual void _update_stress_tensor();

    virtual void _init_metal_md_symbols();
    virtual void _create_compute_pipelines();

public:
    MD_MetalBackend();
    virtual ~MD_MetalBackend();

    virtual void get_settings(input_file &inp) override;
    virtual void init() override;

    virtual void sim_step() override;

    virtual void apply_simulation_data_changes() override;
    virtual void apply_changes_to_simulation_data() override;
};

#endif /* MD_METALBACKEND_H_ */
