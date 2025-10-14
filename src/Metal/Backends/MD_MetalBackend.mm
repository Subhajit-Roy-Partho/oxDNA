/**
 * @file    MD_MetalBackend.mm
 * @brief   Implementation of Metal MD backend
 */

#include "MD_MetalBackend.h"
#include "../../Utilities/oxDNAException.h"

MD_MetalBackend::MD_MetalBackend() :
    MDBackend(),
    MetalBaseBackend(),
    _use_edge(false),
    _any_rigid_body(false),
    _avoid_cpu_calculations(false),
    _first_step_pipeline(nil),
    _second_step_pipeline(nil),
    _forces_pipeline(nil),
    _zero_forces_pipeline(nil),
    _d_vels(nil),
    _d_Ls(nil),
    _d_forces(nil),
    _d_torques(nil),
    _h_vels(nullptr),
    _h_Ls(nullptr),
    _h_forces(nullptr),
    _h_torques(nullptr),
    _d_particles_to_mols(nil),
    _d_mol_sizes(nil),
    _d_molecular_coms(nil),
    _d_buff_vels(nil),
    _d_buff_Ls(nil),
    _d_buff_particles_to_mols(nil),
    _barostat_attempts(0),
    _barostat_accepted(0),
    _update_st_every(0),
    _print_energy(false),
    _obs_output_error_conf(nullptr),
    _d_ext_forces(nil),
    _max_ext_forces(0) {
}

MD_MetalBackend::~MD_MetalBackend() {
    // Release Metal resources
    _first_step_pipeline = nil;
    _second_step_pipeline = nil;
    _forces_pipeline = nil;
    _zero_forces_pipeline = nil;

    _d_vels = nil;
    _d_Ls = nil;
    _d_forces = nil;
    _d_torques = nil;
    _d_particles_to_mols = nil;
    _d_mol_sizes = nil;
    _d_molecular_coms = nil;
    _d_buff_vels = nil;
    _d_buff_Ls = nil;
    _d_buff_particles_to_mols = nil;
    _d_ext_forces = nil;

    if(_h_vels) delete[] _h_vels;
    if(_h_Ls) delete[] _h_Ls;
    if(_h_forces) delete[] _h_forces;
    if(_h_torques) delete[] _h_torques;
}

void MD_MetalBackend::get_settings(input_file &inp) {
    MDBackend::get_settings(inp);
    MetalBaseBackend::get_settings(inp);

    getInputBool(&inp, "use_edge", &_use_edge, 0);
    getInputBool(&inp, "Metal_avoid_cpu_calculations", &_avoid_cpu_calculations, 0);
    getInputInt(&inp, "update_st_every", &_update_st_every, 0);
    getInputBool(&inp, "print_energy_every", &_print_energy, 0);

    if(getInputString(&inp, "error_conf_file", _error_conf_file, 0) == KEY_FOUND) {
        _obs_output_error_conf = new ObservableOutput(_error_conf_file);
    }
}

void MD_MetalBackend::init() {
    @autoreleasepool {
        MDBackend::init();

        // Initialize Metal backend
        MetalBaseBackend::init_metal();

        // Store particle count
        _N = this->_particles.size();

        // Allocate host arrays
        _h_vels = new m_number4[_N];
        _h_Ls = new m_number4[_N];
        _h_forces = new m_number4[_N];
        _h_torques = new m_number4[_N];
        _h_poss = new m_number4[_N];
        _h_bonds = new MetalBonds[_N];
        _h_orientations = new m_quat[_N];

        // Allocate device buffers
        _d_vels = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_Ls = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_forces = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_torques = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_poss = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_bonds = MetalUtils::allocate_buffer<MetalBonds>(_device, _N);
        _d_orientations = MetalUtils::allocate_buffer<m_quat>(_device, _N);
        _d_list_poss = MetalUtils::allocate_buffer<m_number4>(_device, _N);

        // Allocate box buffer
        _d_metal_box = MetalUtils::allocate_buffer<MetalBox>(_device, 1);

        // Initialize particle data from config
        for(int i = 0; i < _N; i++) {
            BaseParticle *p = this->_particles[i];

            _h_poss[i].x = p->pos.x;
            _h_poss[i].y = p->pos.y;
            _h_poss[i].z = p->pos.z;
            _h_poss[i].w = 1;

            _h_vels[i].x = p->vel.x;
            _h_vels[i].y = p->vel.y;
            _h_vels[i].z = p->vel.z;
            _h_vels[i].w = 0.0;

            _h_Ls[i].x = p->L.x;
            _h_Ls[i].y = p->L.y;
            _h_Ls[i].z = p->L.z;
            _h_Ls[i].w = 0.0;

            // Convert orientation matrix to quaternion (simplified)
            // TODO: Proper matrix to quaternion conversion
            _h_orientations[i].x = 0.0;
            _h_orientations[i].y = 0.0;
            _h_orientations[i].z = 0.0;
            _h_orientations[i].w = 1.0;

            // Bonds
            _h_bonds[i].n3 = (p->n3 != P_VIRTUAL) ? p->n3->index : -1;
            _h_bonds[i].n5 = (p->n5 != P_VIRTUAL) ? p->n5->index : -1;
        }

        // Copy to GPU
        _host_to_gpu();

        // Create compute pipelines
        _create_compute_pipelines();

        // Initialize Metal-specific symbols and constants
        _init_metal_md_symbols();

        OX_LOG(Logger::LOG_INFO, "Metal MD Backend initialized with %d particles", _N);
        OX_LOG(Logger::LOG_INFO, "Allocated GPU memory: %.2f MB", MetalUtils::get_allocated_mem_mb());
    }
}

void MD_MetalBackend::_create_compute_pipelines() {
    @autoreleasepool {
        NSError *error = nil;

        // Create first step pipeline
        id<MTLFunction> first_step_func = [_library newFunctionWithName:@"first_step_velocity_verlet"];
        if(!first_step_func) {
            throw oxDNAException("Failed to find first_step_velocity_verlet kernel function");
        }
        _first_step_pipeline = [_device newComputePipelineStateWithFunction:first_step_func error:&error];
        METAL_CHECK_ERROR(_first_step_pipeline, [[error localizedDescription] UTF8String]);

        // Create second step pipeline
        id<MTLFunction> second_step_func = [_library newFunctionWithName:@"second_step_velocity_verlet"];
        if(!second_step_func) {
            throw oxDNAException("Failed to find second_step_velocity_verlet kernel function");
        }
        _second_step_pipeline = [_device newComputePipelineStateWithFunction:second_step_func error:&error];
        METAL_CHECK_ERROR(_second_step_pipeline, [[error localizedDescription] UTF8String]);

        // Create zero forces pipeline
        id<MTLFunction> zero_forces_func = [_library newFunctionWithName:@"zero_forces"];
        if(!zero_forces_func) {
            throw oxDNAException("Failed to find zero_forces kernel function");
        }
        _zero_forces_pipeline = [_device newComputePipelineStateWithFunction:zero_forces_func error:&error];
        METAL_CHECK_ERROR(_zero_forces_pipeline, [[error localizedDescription] UTF8String]);

        OX_LOG(Logger::LOG_INFO, "Metal compute pipelines created successfully");
    }
}

void MD_MetalBackend::_init_metal_md_symbols() {
    // Initialize MD-specific constants and symbols
    // This would include setting up constant buffers for MD parameters
}

void MD_MetalBackend::_host_to_gpu() {
    MetalBaseBackend::_host_to_gpu();

    MetalUtils::copy_to_device<m_number4>(_d_vels, _h_vels, _N);
    MetalUtils::copy_to_device<m_number4>(_d_Ls, _h_Ls, _N);
}

void MD_MetalBackend::_gpu_to_host() {
    MetalBaseBackend::_gpu_to_host();

    MetalUtils::copy_from_device<m_number4>(_h_vels, _d_vels, _N);
    MetalUtils::copy_from_device<m_number4>(_h_Ls, _d_Ls, _N);
    MetalUtils::copy_from_device<m_number4>(_h_forces, _d_forces, _N);
    MetalUtils::copy_from_device<m_number4>(_h_torques, _d_torques, _N);

    // Update particle data
    for(int i = 0; i < _N; i++) {
        BaseParticle *p = this->_particles[i];

        p->pos.x = _h_poss[i].x;
        p->pos.y = _h_poss[i].y;
        p->pos.z = _h_poss[i].z;

        p->vel.x = _h_vels[i].x;
        p->vel.y = _h_vels[i].y;
        p->vel.z = _h_vels[i].z;

        p->L.x = _h_Ls[i].x;
        p->L.y = _h_Ls[i].y;
        p->L.z = _h_Ls[i].z;

        p->force.x = _h_forces[i].x;
        p->force.y = _h_forces[i].y;
        p->force.z = _h_forces[i].z;

        p->torque.x = _h_torques[i].x;
        p->torque.y = _h_torques[i].y;
        p->torque.z = _h_torques[i].z;
    }
}

void MD_MetalBackend::_first_step() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:_first_step_pipeline];

        // Set buffers
        [encoder setBuffer:_d_poss offset:0 atIndex:0];
        [encoder setBuffer:_d_vels offset:0 atIndex:1];
        [encoder setBuffer:_d_forces offset:0 atIndex:2];

        // Set time step parameters
        m_number dt = this->_dt;
        m_number dt_half = dt / 2.0;
        [encoder setBytes:&dt length:sizeof(m_number) atIndex:3];
        [encoder setBytes:&dt_half length:sizeof(m_number) atIndex:4];
        [encoder setBuffer:_d_metal_box offset:0 atIndex:5];

        // Dispatch
        MTLSize gridSize = MTLSizeMake(_N, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(_particles_kernel_cfg.threads_per_threadgroup, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MD_MetalBackend::_forces_second_step() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:_second_step_pipeline];

        // Set buffers
        [encoder setBuffer:_d_poss offset:0 atIndex:0];
        [encoder setBuffer:_d_vels offset:0 atIndex:1];
        [encoder setBuffer:_d_forces offset:0 atIndex:2];

        m_number dt_half = this->_dt / 2.0;
        [encoder setBytes:&dt_half length:sizeof(m_number) atIndex:3];

        // Dispatch
        MTLSize gridSize = MTLSizeMake(_N, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(_particles_kernel_cfg.threads_per_threadgroup, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MD_MetalBackend::sim_step() {
    // Zero forces
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:_zero_forces_pipeline];
        [encoder setBuffer:_d_forces offset:0 atIndex:0];

        MTLSize gridSize = MTLSizeMake(_N, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(_particles_kernel_cfg.threads_per_threadgroup, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    // First step of integration
    _first_step();

    // Compute forces (would call interaction kernels here)
    // TODO: Implement force computation kernels

    // Apply thermostat if needed
    _thermalize();

    // Second step of integration
    _forces_second_step();

    // Update time
    _config_info->curr_step++;
}

void MD_MetalBackend::_thermalize() {
    // TODO: Implement thermostats for Metal
}

void MD_MetalBackend::_apply_barostat() {
    // TODO: Implement barostat for Metal
}

void MD_MetalBackend::_update_stress_tensor() {
    // TODO: Implement stress tensor calculation
}

void MD_MetalBackend::_set_external_forces() {
    // TODO: Implement external forces
}

void MD_MetalBackend::_apply_external_forces_changes() {
    // TODO: Implement external force updates
}

void MD_MetalBackend::_sort_particles() {
    MetalBaseBackend::_sort_index();
}

void MD_MetalBackend::_rescale_positions(m_number4 new_Ls, m_number4 old_Ls) {
    // TODO: Implement position rescaling for NPT
}

void MD_MetalBackend::_rescale_molecular_positions(m_number4 new_Ls, m_number4 old_Ls, bool is_reverse_move) {
    // TODO: Implement molecular position rescaling
}

void MD_MetalBackend::apply_simulation_data_changes() {
    _gpu_to_host();
}

void MD_MetalBackend::apply_changes_to_simulation_data() {
    _host_to_gpu();
}
