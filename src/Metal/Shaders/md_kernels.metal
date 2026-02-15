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
inline m_number4 quat_multiply_md(m_number4 a, m_number4 b) {
    return m_number4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

inline m_number4 update_orientation_from_L(m_number4 L, m_number4 old_orientation, m_number dt) {
    m_number3 Lvec = L.xyz;
    m_number norm = length(Lvec);
    if(norm <= (m_number) 1e-12f) {
        return old_orientation;
    }

    m_number3 axis = Lvec / norm;
    m_number half_theta = dt * norm * (m_number) 0.5f;
    m_number s = sin(half_theta);
    m_number c = cos(half_theta);

    m_number4 R = m_number4(axis.x * s, axis.y * s, axis.z * s, c);
    m_number4 updated = quat_multiply_md(old_orientation, R);

    m_number qnorm = length(updated);
    if(qnorm <= (m_number) 1e-12f) {
        return old_orientation;
    }
    return updated / qnorm;
}

kernel void first_step_velocity_verlet(
    device m_number4 *positions [[buffer(0)]],
    device m_number4 *orientations [[buffer(1)]],
    device m_number4 *velocities [[buffer(2)]],
    device m_number4 *angular_momenta [[buffer(3)]],
    device m_number4 *forces [[buffer(4)]],
    device m_number4 *torques [[buffer(5)]],
    constant MetalBox &box [[buffer(6)]],
    constant m_number &dt [[buffer(7)]],
    constant m_number &dt_half [[buffer(8)]],
    constant int &N [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if((int) gid >= N) {
        return;
    }

    // v += F * dt / (2m)
    m_number4 v = velocities[gid];
    const m_number4 F = forces[gid];
    v.x += F.x * dt_half;
    v.y += F.y * dt_half;
    v.z += F.z * dt_half;

    // r += v * dt
    m_number4 r = positions[gid];
    r.x += v.x * dt;
    r.y += v.y * dt;
    r.z += v.z * dt;

    m_number4 L = angular_momenta[gid];
    const m_number4 T = torques[gid];
    L.x += T.x * dt_half;
    L.y += T.y * dt_half;
    L.z += T.z * dt_half;

    positions[gid] = r;
    velocities[gid] = v;
    angular_momenta[gid] = L;
    orientations[gid] = update_orientation_from_L(L, orientations[gid], dt);
}

/**
 * @brief Update velocities (second half of velocity Verlet)
 */
kernel void second_step_velocity_verlet(
    device m_number4 *velocities [[buffer(0)]],
    device m_number4 *angular_momenta [[buffer(1)]],
    device m_number4 *forces [[buffer(2)]],
    device m_number4 *torques [[buffer(3)]],
    constant m_number &dt_half [[buffer(4)]],
    constant int &N [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if((int) gid >= N) {
        return;
    }

    // v += F * dt / (2m)
    m_number4 v = velocities[gid];
    m_number4 F = forces[gid];
    v.x += F.x * dt_half;
    v.y += F.y * dt_half;
    v.z += F.z * dt_half;
    v.w = (v.x * v.x + v.y * v.y + v.z * v.z) * (m_number) 0.5f;
    velocities[gid] = v;

    m_number4 L = angular_momenta[gid];
    m_number4 T = torques[gid];
    L.x += T.x * dt_half;
    L.y += T.y * dt_half;
    L.z += T.z * dt_half;
    L.w = (L.x * L.x + L.y * L.y + L.z * L.z) * (m_number) 0.5f;
    angular_momenta[gid] = L;
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

    // Quaternion derivative: dq/dt = 0.5 * q * omega_quat (Body frame)
    m_number4 omega_quat = m_number4(omega.x, omega.y, omega.z, 0.0);

    // Quaternion multiplication q * omega
    // Real: -v.w
    // Vec: s*w + v x w
    // q = (x,y,z,w). w=scalar. Note: my q struct has w at end?
    // q.w is scalar part (based get_axes usage).
    // q.x, q.y, q.z is vector part.
    // omega = (Lx, Ly, Lz).
    // dq.x = q.w * Lx + (q.y * Lz - q.z * Ly)
    // dq.y = q.w * Ly + (q.z * Lx - q.x * Lz)
    // dq.z = q.w * Lz + (q.x * Ly - q.y * Lx)
    // dq.w = - (q.x * Lx + q.y * Ly + q.z * Lz)
    
    m_number4 dq;
    dq.x = 0.5 * (q.w * omega.x + q.y * omega.z - q.z * omega.y);
    dq.y = 0.5 * (q.w * omega.y + q.z * omega.x - q.x * omega.z);
    dq.z = 0.5 * (q.w * omega.z + q.x * omega.y - q.y * omega.x);
    dq.w = -0.5 * (q.x * omega.x + q.y * omega.y + q.z * omega.z);

    // Update quaternion
    q += dq * dt;

    // Normalize quaternion
    m_number norm = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    orientations[gid] = q / norm;
}

/**
 * @brief Update angular momenta (L += T * dt)
 */
kernel void update_angular_momenta(
    device m_number4 *angular_momenta [[buffer(0)]],
    device m_number4 *torques [[buffer(1)]],
    constant m_number &dt [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // L += T * dt
    angular_momenta[gid].x += torques[gid].x * dt;
    angular_momenta[gid].y += torques[gid].y * dt;
    angular_momenta[gid].z += torques[gid].z * dt;
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
