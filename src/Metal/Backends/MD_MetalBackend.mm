/**
 * @file    MD_MetalBackend.mm
 * @brief   Implementation of Metal MD backend
 */

#include "MD_MetalBackend.h"

#include "../../Utilities/ConfigInfo.h"
#include "../../Utilities/oxDNAException.h"
#include "../Interactions/MetalInteractionFactory.h"
#include "../Lists/MetalListFactory.h"
#include "../Thermostats/MetalThermostatFactory.h"

#include <cmath>

namespace {

inline m_quat _quat_from_orientation(const LR_matrix &o) {
    m_quat q;

    number trace = o.v1.x + o.v2.y + o.v3.z;
    if(trace > 0) {
        number s = 0.5 / sqrt(trace + 1.0);
        q.w = (m_number) (0.25 / s);
        q.x = (m_number) ((o.v3.y - o.v2.z) * s);
        q.y = (m_number) ((o.v1.z - o.v3.x) * s);
        q.z = (m_number) ((o.v2.x - o.v1.y) * s);
    }
    else {
        if((o.v1.x > o.v2.y) && (o.v1.x > o.v3.z)) {
            number s = 0.5 / sqrt(1.0 + o.v1.x - o.v2.y - o.v3.z);
            q.w = (m_number) ((o.v3.y - o.v2.z) * s);
            q.x = (m_number) (0.25 / s);
            q.y = (m_number) ((o.v1.y + o.v2.x) * s);
            q.z = (m_number) ((o.v1.z + o.v3.x) * s);
        }
        else if(o.v2.y > o.v3.z) {
            number s = 0.5 / sqrt(1.0 + o.v2.y - o.v1.x - o.v3.z);
            q.w = (m_number) ((o.v1.z - o.v3.x) * s);
            q.x = (m_number) ((o.v1.y + o.v2.x) * s);
            q.y = (m_number) (0.25 / s);
            q.z = (m_number) ((o.v2.z + o.v3.y) * s);
        }
        else {
            number s = 0.5 / sqrt(1.0 + o.v3.z - o.v1.x - o.v2.y);
            q.w = (m_number) ((o.v2.x - o.v1.y) * s);
            q.x = (m_number) ((o.v1.z + o.v3.x) * s);
            q.y = (m_number) ((o.v2.z + o.v3.y) * s);
            q.z = (m_number) (0.25 / s);
        }
    }

    return q;
}

inline LR_matrix _orientation_from_quat(const m_quat &q) {
    number sqx = (number) q.x * (number) q.x;
    number sqy = (number) q.y * (number) q.y;
    number sqz = (number) q.z * (number) q.z;
    number sqw = (number) q.w * (number) q.w;
    number xy = (number) q.x * (number) q.y;
    number xz = (number) q.x * (number) q.z;
    number xw = (number) q.x * (number) q.w;
    number yz = (number) q.y * (number) q.z;
    number yw = (number) q.y * (number) q.w;
    number zw = (number) q.z * (number) q.w;
    number norm = sqx + sqy + sqz + sqw;
    if(norm < 1e-12) {
        return LR_matrix((number) 1., (number) 0., (number) 0.,
                         (number) 0., (number) 1., (number) 0.,
                         (number) 0., (number) 0., (number) 1.);
    }
    number invs = 1.0 / norm;

    LR_matrix orientation;
    orientation.v1.x = (sqx - sqy - sqz + sqw) * invs;
    orientation.v1.y = 2 * (xy - zw) * invs;
    orientation.v1.z = 2 * (xz + yw) * invs;
    orientation.v2.x = 2 * (xy + zw) * invs;
    orientation.v2.y = (-sqx + sqy - sqz + sqw) * invs;
    orientation.v2.z = 2 * (yz - xw) * invs;
    orientation.v3.x = 2 * (xz - yw) * invs;
    orientation.v3.y = 2 * (yz + xw) * invs;
    orientation.v3.z = (-sqx - sqy + sqz + sqw) * invs;

    return orientation;
}

}

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
    _zero_torques_pipeline(nil),
    _update_angular_momenta_pipeline(nil),
    _update_orientations_pipeline(nil),
    _d_vels(nil),
    _d_Ls(nil),
    _d_forces(nil),
    _d_torques(nil),
    _h_vels(nullptr),
    _h_Ls(nullptr),
    _h_forces(nullptr),
    _h_torques(nullptr),
    _d_energies(nil),
    _h_energies(nullptr),
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
    _first_step_pipeline = nil;
    _second_step_pipeline = nil;
    _forces_pipeline = nil;
    _zero_forces_pipeline = nil;
    _zero_torques_pipeline = nil;
    _update_angular_momenta_pipeline = nil;
    _update_orientations_pipeline = nil;

    _d_vels = nil;
    _d_Ls = nil;
    _d_forces = nil;
    _d_torques = nil;
    _d_energies = nil;

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
    if(_h_energies) delete[] _h_energies;

    if(_metal_list) delete _metal_list;
    if(_metal_interaction) delete _metal_interaction;
    if(_metal_thermostat) delete _metal_thermostat;
    if(_obs_output_error_conf) delete _obs_output_error_conf;
}

void MD_MetalBackend::get_settings(input_file &inp) {
    MDBackend::get_settings(inp);
    MetalBaseBackend::get_settings(inp);

    getInputBool(&inp, "use_edge", &_use_edge, 0);
    getInputBool(&inp, "Metal_avoid_cpu_calculations", &_avoid_cpu_calculations, 0);
    getInputInt(&inp, "update_st_every", &_update_st_every, 0);

    int print_energy_int = 0;
    if(getInputInt(&inp, "print_energy_every", &print_energy_int, 0) == KEY_FOUND) {
        _print_energy = (print_energy_int > 0);
    }

    if(getInputString(&inp, "error_conf_file", _error_conf_file, 0) == KEY_FOUND) {
        _obs_output_error_conf = new ObservableOutput(_error_conf_file);
    }

    _metal_list = MetalListFactory::make_list(inp);
    _metal_list->get_settings(inp);

    _metal_interaction = MetalInteractionFactory::make_interaction(inp);
    _metal_interaction->get_settings(inp);
    _metal_interaction->get_metal_settings(inp);

    _metal_thermostat = MetalThermostatFactory::make_thermostat(inp);
    if(_metal_thermostat != nullptr) {
        _metal_thermostat->get_settings(inp);
    }
}

void MD_MetalBackend::init() {
    @autoreleasepool {
        MDBackend::init();
        MetalBaseBackend::init_metal();

        _N = (int) this->_particles.size();

        _h_poss = new m_number4[_N];
        _h_vels = new m_number4[_N];
        _h_Ls = new m_number4[_N];
        _h_forces = new m_number4[_N];
        _h_torques = new m_number4[_N];
        _h_orientations = new m_quat[_N];
        _h_bonds = new MetalBonds[_N];
        _h_energies = new float[_N * 10];
        _h_particles_to_mols.resize(_N);

        _d_poss = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_bonds = MetalUtils::allocate_buffer<MetalBonds>(_device, _N);
        _d_orientations = MetalUtils::allocate_buffer<m_quat>(_device, _N);
        _d_list_poss = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_vels = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_Ls = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_forces = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_torques = MetalUtils::allocate_buffer<m_number4>(_device, _N);
        _d_energies = MetalUtils::allocate_buffer<float>(_device, _N * 10);

        _h_metal_box.set_Metal_from_CPU(this->_box.get());
        _d_metal_box = MetalUtils::allocate_buffer<MetalBox::BoxData>(_device, 1);

        _create_compute_pipelines();
        _init_metal_md_symbols();

        _metal_list->metal_init(_N, _rcut, &_h_metal_box, _d_metal_box, _device, _library);
        _metal_interaction->metal_init(_N, _device, _library);
        if(_metal_thermostat != nullptr) {
            _metal_thermostat->metal_init(_N, _device, _library);
        }
        if(_obs_output_error_conf != nullptr) {
            _obs_output_error_conf->init();
        }

        _update_host_buffers_from_particles();
        _host_to_gpu();
        _metal_list->update(_d_poss, _d_list_poss, _d_bonds);
        _zero_force_and_torque_buffers();
        _metal_interaction->compute_forces(_metal_list,
                                           _d_poss,
                                           _d_orientations,
                                           _d_forces,
                                           _d_torques,
                                           _d_bonds,
                                           _d_metal_box,
                                           _d_energies);

        if(_metal_interaction->use_cpu_fallback()) {
            OX_LOG(Logger::LOG_INFO, "Metal interaction mode: CPU fallback (Metal_avoid_cpu_calculations = 0)");
        }
        else {
            OX_LOG(Logger::LOG_INFO, "Metal interaction mode: native kernels (Metal_avoid_cpu_calculations = 1)");
        }

        OX_LOG(Logger::LOG_INFO, "Metal MD Backend initialized with %d particles", _N);
        OX_LOG(Logger::LOG_INFO, "Allocated GPU memory: %.2f MB", MetalUtils::get_allocated_mem_mb());
    }
}

void MD_MetalBackend::_create_compute_pipelines() {
    @autoreleasepool {
        NSError *error = nil;

        id<MTLFunction> first_step_func = [_library newFunctionWithName:@"first_step_velocity_verlet"];
        if(!first_step_func) {
            throw oxDNAException("Failed to find first_step_velocity_verlet kernel function");
        }
        _first_step_pipeline = [_device newComputePipelineStateWithFunction:first_step_func error:&error];
        METAL_CHECK_ERROR(_first_step_pipeline, [[error localizedDescription] UTF8String]);

        id<MTLFunction> second_step_func = [_library newFunctionWithName:@"second_step_velocity_verlet"];
        if(!second_step_func) {
            throw oxDNAException("Failed to find second_step_velocity_verlet kernel function");
        }
        _second_step_pipeline = [_device newComputePipelineStateWithFunction:second_step_func error:&error];
        METAL_CHECK_ERROR(_second_step_pipeline, [[error localizedDescription] UTF8String]);

        id<MTLFunction> zero_forces_func = [_library newFunctionWithName:@"zero_forces"];
        if(!zero_forces_func) {
            throw oxDNAException("Failed to find zero_forces kernel function");
        }
        _zero_forces_pipeline = [_device newComputePipelineStateWithFunction:zero_forces_func error:&error];
        METAL_CHECK_ERROR(_zero_forces_pipeline, [[error localizedDescription] UTF8String]);

        id<MTLFunction> zero_torques_func = [_library newFunctionWithName:@"zero_torques"];
        if(!zero_torques_func) {
            throw oxDNAException("Failed to find zero_torques kernel function");
        }
        _zero_torques_pipeline = [_device newComputePipelineStateWithFunction:zero_torques_func error:&error];
        METAL_CHECK_ERROR(_zero_torques_pipeline, [[error localizedDescription] UTF8String]);
    }
}

void MD_MetalBackend::_init_metal_md_symbols() {
}

void MD_MetalBackend::_update_host_buffers_from_particles() {
    _any_rigid_body = false;
    for(int i = 0; i < _N; i++) {
        BaseParticle *p = this->_particles[i];

        _h_poss[i].x = (m_number) p->pos.x;
        _h_poss[i].y = (m_number) p->pos.y;
        _h_poss[i].z = (m_number) p->pos.z;
        _h_poss[i].w = (m_number) p->type;

        _h_vels[i].x = (m_number) p->vel.x;
        _h_vels[i].y = (m_number) p->vel.y;
        _h_vels[i].z = (m_number) p->vel.z;
        _h_vels[i].w = (m_number) 0.f;

        _h_Ls[i].x = (m_number) p->L.x;
        _h_Ls[i].y = (m_number) p->L.y;
        _h_Ls[i].z = (m_number) p->L.z;
        _h_Ls[i].w = (m_number) 0.f;

        _h_orientations[i] = _quat_from_orientation(p->orientation);

        _h_bonds[i].n3 = (p->n3 != P_VIRTUAL) ? p->n3->index : -1;
        _h_bonds[i].n5 = (p->n5 != P_VIRTUAL) ? p->n5->index : -1;

        _h_particles_to_mols[i] = p->strand_id;

        if(p->is_rigid_body()) {
            _any_rigid_body = true;
        }
    }
}

void MD_MetalBackend::_update_particles_from_host_buffers() {
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

        p->orientation = _orientation_from_quat(_h_orientations[i]);
        p->orientationT = p->orientation.get_transpose();
        p->set_positions();
    }
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

    if(_d_energies != nil) {
        MetalUtils::copy_from_device<float>(_h_energies, _d_energies, _N * 10);
    }

    _update_particles_from_host_buffers();
}

void MD_MetalBackend::_first_step() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:_first_step_pipeline];
        [encoder setBuffer:_d_poss offset:0 atIndex:0];
        [encoder setBuffer:_d_orientations offset:0 atIndex:1];
        [encoder setBuffer:_d_vels offset:0 atIndex:2];
        [encoder setBuffer:_d_Ls offset:0 atIndex:3];
        [encoder setBuffer:_d_forces offset:0 atIndex:4];
        [encoder setBuffer:_d_torques offset:0 atIndex:5];
        [encoder setBuffer:_d_metal_box offset:0 atIndex:6];

        m_number dt = this->_dt;
        m_number dt_half = dt * (m_number) 0.5f;
        int N = _N;
        [encoder setBytes:&dt length:sizeof(m_number) atIndex:7];
        [encoder setBytes:&dt_half length:sizeof(m_number) atIndex:8];
        [encoder setBytes:&N length:sizeof(int) atIndex:9];

        MTLSize gridSize = MTLSizeMake(_N, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(_particles_kernel_cfg.threads_per_threadgroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MD_MetalBackend::_zero_force_and_torque_buffers() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];

        id<MTLComputeCommandEncoder> zeroForcesEncoder = [commandBuffer computeCommandEncoder];
        [zeroForcesEncoder setComputePipelineState:_zero_forces_pipeline];
        [zeroForcesEncoder setBuffer:_d_forces offset:0 atIndex:0];
        [zeroForcesEncoder dispatchThreads:MTLSizeMake(_N, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(_particles_kernel_cfg.threads_per_threadgroup, 1, 1)];
        [zeroForcesEncoder endEncoding];

        id<MTLComputeCommandEncoder> zeroTorquesEncoder = [commandBuffer computeCommandEncoder];
        [zeroTorquesEncoder setComputePipelineState:_zero_torques_pipeline];
        [zeroTorquesEncoder setBuffer:_d_torques offset:0 atIndex:0];
        [zeroTorquesEncoder dispatchThreads:MTLSizeMake(_N, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(_particles_kernel_cfg.threads_per_threadgroup, 1, 1)];
        [zeroTorquesEncoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MD_MetalBackend::_forces_second_step() {
    _metal_interaction->compute_forces(_metal_list,
                                       _d_poss,
                                       _d_orientations,
                                       _d_forces,
                                       _d_torques,
                                       _d_bonds,
                                       _d_metal_box,
                                       _d_energies);

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:_second_step_pipeline];
        [encoder setBuffer:_d_vels offset:0 atIndex:0];
        [encoder setBuffer:_d_Ls offset:0 atIndex:1];
        [encoder setBuffer:_d_forces offset:0 atIndex:2];
        [encoder setBuffer:_d_torques offset:0 atIndex:3];

        m_number dt_half = this->_dt * (m_number) 0.5f;
        int N = _N;
        [encoder setBytes:&dt_half length:sizeof(m_number) atIndex:4];
        [encoder setBytes:&N length:sizeof(int) atIndex:5];

        MTLSize gridSize = MTLSizeMake(_N, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(_particles_kernel_cfg.threads_per_threadgroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MD_MetalBackend::sim_step() {
    _mytimer->resume();

    if(_metal_interaction->use_cpu_fallback()) {
        const number dt = _dt;
        const number dt_half = _dt * (number) 0.5;

        _timer_first_step->resume();
        _gpu_to_host();

        for(auto p : _particles) {
            p->vel += p->force * dt_half;
            LR_vector dr = p->vel * dt;
            p->pos += dr;

            if(_lees_edwards) {
                const LR_vector &L = _box->box_sides();
                int y_new = floor(p->pos.y / L.y);
                int y_old = floor((p->pos.y - dr.y) / L.y);
                if(y_new != y_old) {
                    number delta_x = _shear_rate * L.y * current_step() * dt;
                    delta_x -= floor(delta_x / L.x) * L.x;
                    if(y_new > y_old) {
                        p->pos.x -= delta_x;
                        p->pos.y -= L.y;
                        p->vel.x -= _shear_rate * L.y;
                    }
                    else {
                        p->pos.x += delta_x;
                        p->pos.y += L.y;
                        p->vel.x += _shear_rate * L.y;
                    }
                }
            }

            if(p->is_rigid_body()) {
                p->L += p->torque * dt_half;

                number norm = p->L.module();
                if(norm <= 1e-12) {
                    continue;
                }
                LR_vector LVersor(p->L / norm);

                number sintheta = sin(dt * norm);
                number costheta = cos(dt * norm);
                number olcos = 1. - costheta;

                number xyo = LVersor[0] * LVersor[1] * olcos;
                number xzo = LVersor[0] * LVersor[2] * olcos;
                number yzo = LVersor[1] * LVersor[2] * olcos;
                number xsin = LVersor[0] * sintheta;
                number ysin = LVersor[1] * sintheta;
                number zsin = LVersor[2] * sintheta;

                LR_matrix R(LVersor[0] * LVersor[0] * olcos + costheta,
                            xyo - zsin,
                            xzo + ysin,
                            xyo + zsin,
                            LVersor[1] * LVersor[1] * olcos + costheta,
                            yzo - xsin,
                            xzo - ysin,
                            yzo + xsin,
                            LVersor[2] * LVersor[2] * olcos + costheta);

                p->orientation = p->orientation * R;
                p->orientationT = p->orientation.get_transpose();
                p->set_positions();
            }
        }

        _update_host_buffers_from_particles();
        _host_to_gpu();
        _timer_first_step->pause();

        _timer_lists->resume();
        _N_updates++;
        _timer_lists->pause();

        _timer_forces->resume();
        _zero_force_and_torque_buffers();
        _metal_interaction->compute_forces(_metal_list,
                                           _d_poss,
                                           _d_orientations,
                                           _d_forces,
                                           _d_torques,
                                           _d_bonds,
                                           _d_metal_box,
                                           _d_energies);

        for(auto p : _particles) {
            p->vel += p->force * dt_half;
            if(p->is_rigid_body()) {
                p->L += p->torque * dt_half;
            }
        }

        _update_host_buffers_from_particles();
        _host_to_gpu();
        _timer_forces->pause();

        _timer_thermostat->resume();
        _thermalize();
        _timer_thermostat->pause();
    }
    else {
        _timer_first_step->resume();
        _first_step();
        _timer_first_step->pause();

        _timer_lists->resume();
        _metal_list->update(_d_poss, _d_list_poss, _d_bonds);
        _N_updates++;
        _timer_lists->pause();

        _timer_forces->resume();
        _zero_force_and_torque_buffers();
        _forces_second_step();
        _timer_forces->pause();

        _timer_thermostat->resume();
        _thermalize();
        _timer_thermostat->pause();
    }

    _mytimer->pause();
}

void MD_MetalBackend::_thermalize() {
    if(_metal_thermostat != nullptr) {
        _metal_thermostat->apply(_d_vels, _d_Ls, _d_orientations, _d_forces, _d_torques, _d_poss);
    }
}

void MD_MetalBackend::_apply_barostat() {
}

void MD_MetalBackend::_update_stress_tensor() {
}

void MD_MetalBackend::_set_external_forces() {
}

void MD_MetalBackend::_apply_external_forces_changes() {
}

void MD_MetalBackend::_sort_particles() {
    MetalBaseBackend::_sort_index();
}

void MD_MetalBackend::_rescale_positions(m_number4 new_Ls, m_number4 old_Ls) {
    (void) new_Ls;
    (void) old_Ls;
}

void MD_MetalBackend::_rescale_molecular_positions(m_number4 new_Ls, m_number4 old_Ls, bool is_reverse_move) {
    (void) new_Ls;
    (void) old_Ls;
    (void) is_reverse_move;
}

void MD_MetalBackend::apply_simulation_data_changes() {
    _gpu_to_host();
}

void MD_MetalBackend::apply_changes_to_simulation_data() {
    _update_host_buffers_from_particles();
    _h_metal_box.set_Metal_from_CPU(this->_box.get());
    _host_to_gpu();
}
