/**
 * @file    MD_MetalBackend.mm
 * @brief   Implementation of Metal MD backend
 */

#include "MD_MetalBackend.h"
#include "../../Utilities/oxDNAException.h"
#include "../Lists/MetalListFactory.h"
#include "../Interactions/MetalInteractionFactory.h"
#include "../Thermostats/MetalThermostatFactory.h"

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
    _max_ext_forces(0),
    _metal_list(nullptr),
    _metal_interaction(nullptr),
    _metal_thermostat(nullptr) {
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
    // Fixed type for print_energy_every to int but controlling boolean _print_energy
    int print_energy_int = 0;
    if(getInputInt(&inp, "print_energy_every", &print_energy_int, 0) == KEY_FOUND) {
        _print_energy = (print_energy_int > 0);
    }

    if(getInputString(&inp, "error_conf_file", _error_conf_file, 0) == KEY_FOUND) {
        _obs_output_error_conf = new ObservableOutput(_error_conf_file);
    }
    
    // Create components
    _metal_list = MetalListFactory::make_list(inp);
    _metal_list->get_settings(inp);
    
    _metal_interaction = MetalInteractionFactory::make_interaction(inp);
    
    _metal_thermostat = MetalThermostatFactory::make_thermostat(inp);
    // _thermostat not in SimBackend, managed manually or added to MDBackend?
    // MDBackend has _timer_thermostat but no _thermostat pointer in base?
    // We'll manage _metal_thermostat lifecycle here.
}

void MD_MetalBackend::init() {
    @autoreleasepool {
        MDBackend::init();

        // Initialize Metal backend
        MetalBaseBackend::init_metal();

        // Store particle count
        _N = this->_particles.size();

        // Allocate host arrays
        _h_poss = new m_number4[_N];
        _h_vels = new m_number4[_N];
        _h_Ls = new m_number4[_N];
        _h_forces = new m_number4[_N];
        _h_torques = new m_number4[_N];
        
        // 10 components per particle
        _h_energies = new float[_N * 10]; 
        _d_energies = MetalUtils::allocate_buffer<float>(_device, _N * 10);
        
        _h_particles_to_mols.resize(_N);
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

        // Initialize MetalBox from SimulationBox
        _h_metal_box.set_Metal_from_CPU(this->_box.get());

        // Allocate box buffer (using BoxData struct to match shader layout)
        _d_metal_box = MetalUtils::allocate_buffer<MetalBox::BoxData>(_device, 1);
        
        // Copy box data to GPU immediately so List init can use it
        MetalBox::BoxData box_data = _h_metal_box.get_box_data();
        // MetalUtils::copy_to_device(_d_metal_box, &box_data, 1);
        // Direct memcpy to ensure data is there (Safe for Shared mode)
        memcpy(_d_metal_box.contents, &box_data, sizeof(MetalBox::BoxData));
        
        // Initialize Metal List PSOs and buffers
        
        // Initialize Metal List PSOs and buffers
        _metal_list->metal_init(_N, this->_interaction->get_rcut(), &_h_metal_box, _d_metal_box, _device, _library);

        // Initialize particle data from config
        for(int i = 0; i < _N; i++) {
            BaseParticle *p = this->_particles[i];

            _h_poss[i].x = p->pos.x;
            _h_poss[i].y = p->pos.y;
            _h_poss[i].z = p->pos.z;
            _h_poss[i].w = (float)p->type;

            _h_vels[i].x = p->vel.x;
            _h_vels[i].y = p->vel.y;
            _h_vels[i].z = p->vel.z;
            _h_vels[i].w = 0.0;

            _h_Ls[i].x = p->L.x;
            _h_Ls[i].y = p->L.y;
            _h_Ls[i].z = p->L.z;
            _h_Ls[i].w = 0.0;

            // Convert orientation matrix to quaternion
            // p->orientationT has columns v1, v2, v3 (or rows? BaseParticle.h says orientationT is transpose)
            // v1, v2, v3 are stored as vectors.
            // Using standard conversion algorithm.
            // Assuming orientationT columns are the axes.
            LR_vector v1 = p->orientationT.v1;
            LR_vector v2 = p->orientationT.v2;
            LR_vector v3 = p->orientationT.v3;
            
            double trace = v1.x + v2.y + v3.z;
            double qw, qx, qy, qz;
            
            if (trace > 0) {
                double S = 0.5 / sqrt(trace + 1.0);
                qw = 0.25 / S;
                qx = (v2.z - v3.y) * S;
                qy = (v3.x - v1.z) * S;
                qz = (v1.y - v2.x) * S;
            } else {
                if (v1.x > v2.y && v1.x > v3.z) {
                    double S = 2.0 * sqrt(1.0 + v1.x - v2.y - v3.z);
                    qw = (v2.z - v3.y) / S;
                    qx = 0.25 * S;
                    qy = (v1.y + v2.x) / S;
                    qz = (v1.z + v3.x) / S;
                } else if (v2.y > v3.z) {
                    double S = 2.0 * sqrt(1.0 + v2.y - v1.x - v3.z);
                    qw = (v3.x - v1.z) / S;
                    qx = (v1.y + v2.x) / S;
                    qy = 0.25 * S;
                    qz = (v2.z + v3.y) / S;
                } else {
                    double S = 2.0 * sqrt(1.0 + v3.z - v1.x - v2.y);
                    qw = (v1.y - v2.x) / S;
                    qx = (v1.z + v3.x) / S;
                    qy = (v2.z + v3.y) / S;
                    qz = 0.25 * S;
                }
            }
            
            _h_orientations[i].x = (m_number)qx;
            _h_orientations[i].y = (m_number)qy;
            _h_orientations[i].z = (m_number)qz;
            _h_orientations[i].w = (m_number)qw;

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
        
        // Initialize components
        _metal_list->metal_init(_N, _rcut, &_h_metal_box, _d_metal_box, _device, _library);
        _metal_interaction->metal_init(_N, _device, _library);
        if(_metal_thermostat) _metal_thermostat->metal_init(_N, _device, _library);
        
        // Initial update
        _metal_list->update(_d_poss, _d_list_poss, _d_bonds);

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
        
        // Create angular update pipelines
        id<MTLFunction> update_L_func = [_library newFunctionWithName:@"update_angular_momenta"];
        if(update_L_func) {
             _update_angular_momenta_pipeline = [_device newComputePipelineStateWithFunction:update_L_func error:&error];
             METAL_CHECK_ERROR(_update_angular_momenta_pipeline, [[error localizedDescription] UTF8String]);
        }
        
        id<MTLFunction> update_q_func = [_library newFunctionWithName:@"update_orientations"];
        if(update_q_func) {
             _update_orientations_pipeline = [_device newComputePipelineStateWithFunction:update_q_func error:&error];
             METAL_CHECK_ERROR(_update_orientations_pipeline, [[error localizedDescription] UTF8String]);
        }

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
    MetalUtils::copy_to_device<m_number4>(_d_vels, _h_vels, _N);
    MetalUtils::copy_to_device<m_number4>(_d_Ls, _h_Ls, _N);
    
    // Add missing copies
    MetalUtils::copy_to_device<m_number4>(_d_poss, _h_poss, _N);
    MetalUtils::copy_to_device<m_quat>(_d_orientations, _h_orientations, _N);
    MetalUtils::copy_to_device<MetalBonds>(_d_bonds, _h_bonds, _N);
    
    // Copy box
    MetalBox::BoxData box_data = _h_metal_box.get_box_data();
    printf("DEBUG: Copying box to GPU: sides [%f %f %f]\n", box_data.box_sides[0], box_data.box_sides[1], box_data.box_sides[2]);
    MetalUtils::copy_to_device<MetalBox::BoxData>(_d_metal_box, &box_data, 1);
}

void MD_MetalBackend::_gpu_to_host() {
    MetalBaseBackend::_gpu_to_host();

    MetalUtils::copy_from_device<m_number4>(_h_vels, _d_vels, _N);
    MetalUtils::copy_from_device<m_number4>(_h_Ls, _d_Ls, _N);
    MetalUtils::copy_from_device<m_number4>(_h_forces, _d_forces, _N);
    MetalUtils::copy_from_device<m_number4>(_h_torques, _d_torques, _N);
    MetalUtils::copy_from_device<float>(_h_energies, _d_energies, _N * 10);

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
    // Debug: print particle 0 info before first step
    if(_config_info->curr_step < 5) {
        MetalUtils::copy_from_device<m_number4>(_h_poss, _d_poss, _N);
        MetalUtils::copy_from_device<m_number4>(_h_vels, _d_vels, _N);
        MetalUtils::copy_from_device<m_number4>(_h_forces, _d_forces, _N);
        printf("Step %lld - Pre-FirstStep - P0: pos=(%f,%f,%f) vel=(%f,%f,%f) force=(%f,%f,%f) dt=%f\n",
               _config_info->curr_step,
               (float)_h_poss[0].x, (float)_h_poss[0].y, (float)_h_poss[0].z,
               (float)_h_vels[0].x, (float)_h_vels[0].y, (float)_h_vels[0].z,
               (float)_h_forces[0].x, (float)_h_forces[0].y, (float)_h_forces[0].z,
               (float)this->_dt);
    }

    _first_step();

    // Compute forces (would call interaction kernels here)
    // TODO: Implement force computation kernels
    // Zero forces handled in sim_step start
    
    // Update neighbors if needed
    // _metal_list->update handled automatically? No, needs check.
    // BaseList::is_updated() logic?
    // For simple verlet, update every step or check sort?
    // MetalSimpleVerletList has is_updated? 
    // We'll force update for now or implement check logic later
    _metal_list->update(_d_poss, _d_list_poss, _d_bonds); // Fixed arguments
    
    _metal_interaction->compute_forces(_metal_list, _d_poss, _d_orientations, _d_forces, _d_torques, _d_bonds, _d_metal_box, _d_energies);

    // Log energies every 100 steps or first few steps
    if(_config_info->curr_step < 10 || _config_info->curr_step % 100 == 0) {
        printf("DEBUG: Copying energies. h=%p d=%p N=%d\n", _h_energies, _d_energies, _N);
        MetalUtils::copy_from_device<m_number4>(_h_forces, _d_forces, _N);
        
        if(_d_energies) {
             MetalUtils::copy_from_device<float>(_h_energies, _d_energies, _N * 10);
        } else {
             printf("DEBUG: _d_energies is NULL!\n");
        }
        
        // Sum energies
        double tot_energies[10] = {0.0};
        for(int i=0; i<_N; i++) {
             for(int k=0; k<10; k++) tot_energies[k] += _h_energies[i*10 + k];
        }
        printf("Step %lld Energies: FENE=%e EXCL=%e STACK=%e\n",
              _config_info->curr_step, tot_energies[0], tot_energies[1], tot_energies[2]);
    }

    // Apply thermostat if needed
    _thermalize();

    // Second step of integration
    _forces_second_step();

    // Update time
    _config_info->curr_step++;
}

void MD_MetalBackend::_thermalize() {
    if(_metal_thermostat) {
        _metal_thermostat->apply(_d_vels, _d_Ls, _d_orientations, _d_forces, _d_torques, _d_poss);
    }
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
