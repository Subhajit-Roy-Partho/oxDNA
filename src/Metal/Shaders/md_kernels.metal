/**
 * @file    md_kernels.metal
 * @brief   Metal kernels for molecular dynamics
 *
 * Core MD kernels equivalent to CUDA MD kernels
 */

#include <metal_stdlib>
#include "common.metal"
using namespace metal;

/**
 * @brief Update particle velocities (first half of velocity Verlet)
 */
kernel void first_step_velocity_verlet(
    device m_number4 *positions [[buffer(0)]],
    device m_number4 *velocities [[buffer(1)]],
    device m_number4 *forces [[buffer(2)]],
    constant m_number &dt [[buffer(3)]],
    constant m_number &dt_half [[buffer(4)]],
    constant MetalBox &box [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    // Get mass from w component of position
    m_number mass = positions[gid].w;
    m_number inv_mass = 1.0 / mass;

    // v += F * dt / (2m)
    velocities[gid].x += forces[gid].x * dt_half * inv_mass;
    velocities[gid].y += forces[gid].y * dt_half * inv_mass;
    velocities[gid].z += forces[gid].z * dt_half * inv_mass;

    // r += v * dt
    positions[gid].x += velocities[gid].x * dt;
    positions[gid].y += velocities[gid].y * dt;
    positions[gid].z += velocities[gid].z * dt;

    // Apply periodic boundary conditions
    for(int i = 0; i < 3; i++) {
        m_number pos_component = (i == 0) ? positions[gid].x :
                                 (i == 1) ? positions[gid].y :
                                            positions[gid].z;
        pos_component -= box.box_sides[i] * rint(pos_component * box.inv_sides[i]);

        if(i == 0) positions[gid].x = pos_component;
        else if(i == 1) positions[gid].y = pos_component;
        else positions[gid].z = pos_component;
    }
}

/**
 * @brief Update velocities (second half of velocity Verlet)
 */
kernel void second_step_velocity_verlet(
    device m_number4 *positions [[buffer(0)]],
    device m_number4 *velocities [[buffer(1)]],
    device m_number4 *forces [[buffer(2)]],
    constant m_number &dt_half [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    m_number mass = positions[gid].w;
    m_number inv_mass = 1.0 / mass;

    // v += F * dt / (2m)
    velocities[gid].x += forces[gid].x * dt_half * inv_mass;
    velocities[gid].y += forces[gid].y * dt_half * inv_mass;
    velocities[gid].z += forces[gid].z * dt_half * inv_mass;
}

/**
 * @brief Update orientations using quaternions
 */
kernel void update_orientations(
    device m_number4 *orientations [[buffer(0)]],
    device m_number4 *angular_velocities [[buffer(1)]],
    device m_number4 *torques [[buffer(2)]],
    constant m_number &dt [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Get current orientation quaternion
    m_number4 q = orientations[gid];

    // Get angular velocity
    m_number3 omega = m_number3(angular_velocities[gid].x,
                                angular_velocities[gid].y,
                                angular_velocities[gid].z);

    // Quaternion derivative: dq/dt = 0.5 * omega_quat * q
    m_number4 omega_quat = m_number4(omega.x, omega.y, omega.z, 0.0);

    // Quaternion multiplication (simplified for omega_quat.w = 0)
    m_number4 dq;
    dq.x = 0.5 * (omega_quat.w * q.x + omega_quat.x * q.w + omega_quat.y * q.z - omega_quat.z * q.y);
    dq.y = 0.5 * (omega_quat.w * q.y - omega_quat.x * q.z + omega_quat.y * q.w + omega_quat.z * q.x);
    dq.z = 0.5 * (omega_quat.w * q.z + omega_quat.x * q.y - omega_quat.y * q.x + omega_quat.z * q.w);
    dq.w = 0.5 * (omega_quat.w * q.w - omega_quat.x * q.x - omega_quat.y * q.y - omega_quat.z * q.z);

    // Update quaternion
    q += dq * dt;

    // Normalize quaternion
    m_number norm = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    orientations[gid] = q / norm;
}

/**
 * @brief Compute kinetic energy
 */
kernel void compute_kinetic_energy(
    device m_number4 *velocities [[buffer(0)]],
    device m_number4 *positions [[buffer(1)]],
    device m_number *kinetic_energy [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    m_number mass = positions[gid].w;
    m_number3 v = m_number3(velocities[gid].x, velocities[gid].y, velocities[gid].z);
    m_number v_sqr = dot(v, v);

    kinetic_energy[gid] = 0.5 * mass * v_sqr;
}

/**
 * @brief Zero forces array
 */
kernel void zero_forces(
    device m_number4 *forces [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    forces[gid] = m_number4(0.0, 0.0, 0.0, 0.0);
}

/**
 * @brief Zero torques array
 */
kernel void zero_torques(
    device m_number4 *torques [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    torques[gid] = m_number4(0.0, 0.0, 0.0, 0.0);
}

/**
 * @brief Copy buffer
 */
kernel void copy_buffer_m_number4(
    device m_number4 *dest [[buffer(0)]],
    device m_number4 *src [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    dest[gid] = src[gid];
}

/**
 * @brief Reduction sum for m_number4
 */
kernel void reduce_sum_m_number4(
    device m_number4 *input [[buffer(0)]],
    device m_number4 *output [[buffer(1)]],
    threadgroup m_number4 *shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    // Load input into shared memory
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for(uint s = threads / 2; s > 0; s >>= 1) {
        if(tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if(tid == 0) {
        output[0] = shared[0];
    }
}
