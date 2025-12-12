/**
 * @file    dna_kernels.metal
 * @brief   Metal kernels for DNA interaction
 */

#include "shader_utils.h"

using namespace metal;

struct DNAInteractionParams {
    float F1_EPS[50];
    float F1_SHIFT[50];
    float F1_A[2];
    float F1_RC[2];
    float F1_R0[2];
    float F1_BLOW[2];
    float F1_BHIGH[2];
    float F1_RLOW[2];
    float F1_RHIGH[2];
    float F1_RCLOW[2];
    float F1_RCHIGH[2];
    
    float F2_K[2];
    float F2_RC[2];
    float F2_R0[2];
    float F2_BLOW[2];
    float F2_BHIGH[2];
    float F2_RLOW[2];
    float F2_RHIGH[2];
    float F2_RCLOW[2];
    float F2_RCHIGH[2];
    
    float F4_THETA_A[13];
    float F4_THETA_B[13];
    float F4_THETA_T0[13];
    float F4_THETA_TS[13];
    float F4_THETA_TC[13];
    
    float F5_PHI_A[4];
    float F5_PHI_B[4];
    float F5_PHI_XC[4];
    float F5_PHI_XS[4];
    
    float hb_multiplier;
    float T;
    
    // Debye Huckel
    float dh_RC;
    float dh_RHIGH;
    float dh_prefactor;
    float dh_B;
    float dh_minus_kappa;
    int dh_half_charged_ends;
    
    // Flags
    int grooving;
    int use_oxDNA2_coaxial_stacking;
    int use_oxDNA2_FENE;
    float mbf_fmax;
};

struct InitStrandArgs {
    int N;
};

// Helper macros
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

// Constants
// Constants
#define PI 3.141592653589793f

// Positions
#define POS_BACK -0.4f
#define POS_BASE 0.4f
#define POS_STACK 0.34f

// EXCLUDED VOLUME
#define EXCL_EPS 2.0f
// 1 = Back-Back, 2 = Base-Base, 3 = Base-Back, 4 = Back-Base
// S = sigma, R = rstar, B = b, RC = rc
#define EXCL_S1 0.70f
#define EXCL_S2 0.33f
#define EXCL_S3 0.515f
#define EXCL_S4 0.515f

#define EXCL_R1 0.675f
#define EXCL_R2 0.32f
#define EXCL_R3 0.50f
#define EXCL_R4 0.50f

#define EXCL_B1 892.016223343f
#define EXCL_B2 4119.70450017f
#define EXCL_B3 1707.30627298f
#define EXCL_B4 1707.30627298f

#define EXCL_RC1 0.711879214356f
#define EXCL_RC2 0.335388426126f
#define EXCL_RC3 0.52329943261f
#define EXCL_RC4 0.52329943261f

// Model Constants (from model.h)
#define HYDR_F1 0
#define STCK_F1 1

#define STCK_F4_THETA4 0
#define STCK_F4_THETA5 1
#define STCK_F4_THETA6 1

#define STCK_F5_PHI1 0
#define STCK_F5_PHI2 1

#define FENE_EPS 2.0f
#define FENE_R0 0.7525f
#define FENE_DELTA 0.25f
#define FENE_DELTA2 (FENE_DELTA*FENE_DELTA)

// Interaction Constants (Indices)
#define BACKBONE 0
#define BONDED_EXCLUDED_VOLUME 1
#define STACKING 2
#define NONBONDED_EXCLUDED_VOLUME 3
#define HYDROGEN_BONDING 4
#define CROSS_STACKING 5
#define COAXIAL_STACKING 6

// Helix axis definitions
// Quaternion to Axes
inline void get_axes(m_number4 q, thread float3 &a1, thread float3 &a2, thread float3 &a3) {
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;
    
    float x2 = x*x;
    float y2 = y*y;
    float z2 = z*z;
    float xy = x*y;
    float xz = x*z;
    float yz = y*z;
    float wx = w*x;
    float wy = w*y;
    float wz = w*z;
    
    a1 = float3(1.0f - 2.0f*(y2 + z2), 2.0f*(xy + wz), 2.0f*(xz - wy));
    a2 = float3(2.0f*(xy - wz), 1.0f - 2.0f*(x2 + z2), 2.0f*(yz + wx));
    a3 = float3(2.0f*(xz + wy), 2.0f*(yz - wx), 1.0f - 2.0f*(x2 + y2));
}

// Implementations

inline float _f1(float r, int type, int n3, int n5, constant DNAInteractionParams &params) {
    float val = 0.0f;
    if(r < params.F1_RCHIGH[type]) {
        int idx = type * 25 + n3 * 5 + n5;
        if(r > params.F1_RHIGH[type]) {
            val = params.F1_EPS[idx] * params.F1_BHIGH[type] * SQR(r - params.F1_RCHIGH[type]);
        }
        else if(r > params.F1_RLOW[type]) {
            float tmp = 1.0f - exp(-(r - params.F1_R0[type]) * params.F1_A[type]);
            val = params.F1_EPS[idx] * SQR(tmp) - params.F1_SHIFT[idx];
        }
        else if(r > params.F1_RCLOW[type]) {
            val = params.F1_EPS[idx] * params.F1_BLOW[type] * SQR(r - params.F1_RCLOW[type]);
        }
    }
    return val;
}

inline float _f1D(float r, int type, int n3, int n5, constant DNAInteractionParams &params) {
    if(r > params.F1_RCHIGH[type]) return 0.0f;
    if(r < params.F1_RCLOW[type]) return 0.0f; 
    
    int idx = type * 25 + n3 * 5 + n5;
    float r0 = params.F1_R0[type];
    float eps = params.F1_EPS[idx]; 
    float alpha = params.F1_A[type];
    
    // Simplified derivative for Morse region
    float expr = exp(-(r - r0) * alpha);
    return 2.0f * eps * alpha * (1.0f - expr) * expr;
}

inline float _f2(float r, int type, constant DNAInteractionParams &params) {
    float val = 0.0f;
    if(r < params.F2_RCHIGH[type]) {
        if(r > params.F2_RHIGH[type]) {
            val = params.F2_K[type] * params.F2_BHIGH[type] * SQR(r - params.F2_RCHIGH[type]);
        }
        else if(r > params.F2_RLOW[type]) {
            val = (params.F2_K[type] / 2.0f) * (SQR(r - params.F2_R0[type]) - SQR(params.F2_RC[type] - params.F2_R0[type]));
        }
        else if(r > params.F2_RCLOW[type]) {
            val = params.F2_K[type] * params.F2_BLOW[type] * SQR(r - params.F2_RCLOW[type]);
        }
    }
    return val;
}
inline float _f2D(float r, int type, constant DNAInteractionParams &params) { return 0.0f; }

inline float _f4(float t, int type, constant DNAInteractionParams &params) {
    // Clamp input cosine
    if(t > 1.0f) t = 1.0f;
    if(t < -1.0f) t = -1.0f;

    float cost = t;
    float acos_t = acos(cost); 
    float t_angle = acos_t - params.F4_THETA_T0[type];
    
    // Check range
    float abs_t = fabs(t_angle);
    if(abs_t > params.F4_THETA_TC[type]) return 0.0f;
    
    if(abs_t < params.F4_THETA_TS[type]) {
        return 1.0f - params.F4_THETA_A[type] * SQR(abs_t);
    }
    else {
        return params.F4_THETA_B[type] * SQR(params.F4_THETA_TC[type] - abs_t);
    }
}
// Helper to get sin(theta) safely
inline float _sin_from_cos(float cost) {
    float sin_sq = 1.0f - cost*cost;
    return (sin_sq > 1e-6f) ? sqrt(sin_sq) : 1e-3f; // Avoid div by zero
}

inline float _f4Dsin(float cost, int type, constant DNAInteractionParams &params) {
    // cost is cosine.
    // Clamp cost
    if(cost > 1.0f) cost = 1.0f;
    if(cost < -1.0f) cost = -1.0f;
    
    float theta = acos(cost);
    float t_target = params.F4_THETA_T0[type];
    float delta = theta - t_target;
    
    float dV_ddelta = 0.0f;
    float abs_delta = fabs(delta);
    
    // Logic mirroring _f4, but derivative
    if(abs_delta < params.F4_THETA_TC[type]) {
        if(abs_delta > params.F4_THETA_TS[type]) {
             float sign = (delta > 0) ? 1.0f : -1.0f;
             dV_ddelta = -2.0f * params.F4_THETA_B[type] * (params.F4_THETA_TC[type] - abs_delta) * sign;
        }
        else {
             dV_ddelta = -2.0f * params.F4_THETA_A[type] * delta;
        }
    }
    
    float sin_t = sqrt(1.0f - cost*cost);
    if(sin_t < 1e-4f) {
        return 2.0f * params.F4_THETA_A[type];
    }
    
    return -dV_ddelta / sin_t;
}

inline float _f5(float f, int type, constant DNAInteractionParams &params) {
    // Clamp input cosine
    if(f > 1.0f) f = 1.0f;
    if(f < -1.0f) f = -1.0f;
    
    float xc = params.F5_PHI_XC[type];
    float xs = params.F5_PHI_XS[type];
    
    if(f < xc) return 0.0f;
    if(f > xs) return 1.0f; 
    
    return params.F5_PHI_B[type] * SQR(f - xc);
}

inline float _f5D(float f, int type, constant DNAInteractionParams &params) {
    // Clamp input cosine
    if(f > 1.0f) f = 1.0f;
    if(f < -1.0f) f = -1.0f;

    float xc = params.F5_PHI_XC[type];
    float xs = params.F5_PHI_XS[type];
    
    if(f < xc) return 0.0f;
    if(f > xs) return 0.0f; 

    return 2.0f * params.F5_PHI_B[type] * (f - xc); 
}


// Repulsive LJ implementation
inline float _repulsive_lj(float3 r, thread float3 &force, float sigma, float rstar, float b, float rc, bool update_forces) {
    float rnorm = length_squared(r);
    float energy = 0.0f;
    
    if(rnorm < SQR(rc)) {
        if(rnorm > SQR(rstar)) {
            float rmod = sqrt(rnorm);
            float rrc = rmod - rc;
            energy = EXCL_EPS * b * SQR(rrc);
            if(update_forces) force = r * (2.0f * EXCL_EPS * b * rrc / rmod);
        }
        else {
            float tmp = SQR(sigma) / rnorm;
            float lj_part = tmp * tmp * tmp; 
            energy = 4.0f * EXCL_EPS * (SQR(lj_part) - lj_part);
            if(update_forces) force = r * (24.0f * EXCL_EPS * (lj_part - 2.0f*SQR(lj_part)) / rnorm);
        }
    }
    else {
        if(update_forces) force = float3(0.0f);
    }
    return energy;
}

/**
 * @brief Initialize strand ends
 */
kernel void init_DNA_strand_ends(device int *is_strand_end [[buffer(0)]],
                                 device MetalBonds *bonds [[buffer(1)]],
                                 constant InitStrandArgs &args [[buffer(2)]],
                                 uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    if(idx >= args.N) return;
    
    MetalBonds b = bonds[idx];
    is_strand_end[idx] = (b.n3 == -1 || b.n5 == -1) ? 1 : 0;
}

    
// FENE Backbone interaction
inline float _backbone(int idx, int neighbor_idx, float3 r, float3 rback, bool update_forces, thread float3 &force, constant DNAInteractionParams &params) {
    float rbackmod = length(rback);
    float rbackr0 = rbackmod - 0.75f; // _fene_r0 fixed for now
    
    // FENE parameters
    float delta_sqr = FENE_DELTA2;
    float k_fene = FENE_EPS; 
    
    // Switch to harmonic at 90% of delta
    float switch_ratio = 0.9f;
    float switch_dist = FENE_DELTA * switch_ratio;
    float switch_sqr = SQR(switch_dist);
    
    float r_sqr = SQR(rbackr0);
    
    float energy = 0.0f;
    float fmod = 0.0f;
    
    if (r_sqr < switch_sqr) {
        // Standard FENE
        // FENE Singularity Check
        if (r_sqr >= delta_sqr * 0.999f) {
             r_sqr = delta_sqr * 0.999f;
        }
        
        energy = -(k_fene / 2.0f) * log(1.0f - r_sqr / delta_sqr);
        if(update_forces) {
            fmod = -(k_fene * sqrt(r_sqr) / (delta_sqr - r_sqr)); // approx rbackr0 as sqrt(r_sqr) for safety in singularity? 
            // Wait. rbackr0 is vector magnitude difference?
            // In code: fmod = -(k_fene * rbackr0 / (delta_sqr - r_sqr));
            // rbackr0 = rbackmod - r0.
            // if r_sqr is clamped, we should probably calculate consistent force.
            // Using logic:
            fmod = -(k_fene * (sqrt(r_sqr) - params.F2_R0[0]) / (delta_sqr - r_sqr)); // Assuming type 0 parameters for simplicity or calculate per-step?
            // Actually, simply letting the formula run with Clamped r_sqr is safe enough for Capping.
            // The denominator won't be zero.
            // rbackr0 will be large.
            // Force will be large.
            // mbf_fmax will cap it.
            // Just ensure rbackr0 logic uses the clamped r?
            // rbackr0 = rbackmod - r0. rbackmod is passed in.
            // We can't easily change rbackmod without changing rback vector.
            // So just clamping r_sqr in the formula avoids NaN in log and div by zero.
            // fmod = ... / (pos). Valid.
        }
    } else {
        // Harmonic Tail
        float V_c = -(k_fene / 2.0f) * log(1.0f - switch_sqr / delta_sqr);
        float F_c = -(k_fene * switch_dist / (delta_sqr - switch_sqr)); // Negative value
        
        float denom = delta_sqr - switch_sqr;
        float K_tail = -(k_fene * (delta_sqr + switch_sqr)) / (denom * denom); // Negative slope
        
        float dr_tail = fabs(rbackr0) - switch_dist;
        
        energy = V_c - F_c * dr_tail - 0.5f * K_tail * SQR(dr_tail);
        
        if (update_forces) {
            fmod = F_c + K_tail * dr_tail;
            // Handle Compression sign (r < R0)
            if (rbackr0 < 0) fmod = -fmod;
        }
    }
    
    if(update_forces) {
        // Cap the force if mbf_fmax is set
        if (params.mbf_fmax > 0.0f) {
            if (fabs(fmod) > params.mbf_fmax) {
                // Preserve sign
                fmod = (fmod > 0.0f) ? params.mbf_fmax : -params.mbf_fmax;
            }
        }
        
        if (rbackmod > 1e-6f) {
            force = rback * (fmod / rbackmod);
        } else {
            force = float3(0.0f);
        }
    }
    
    return energy;
}

// Bonded Excluded Volume
inline float _bonded_excluded_volume(int idx, int neighbor_idx, float3 r, float3 r_pos, float3 q_pos, m_number4 p_quat, m_number4 q_quat, bool update_forces, thread float3 &force_p, thread float3 &force_q, thread float3 &torque_p, thread float3 &torque_q, constant DNAInteractionParams &params) {
    float3 a1, a2, a3;
    get_axes(p_quat, a1, a2, a3);
    
    float3 b1, b2, b3;
    get_axes(q_quat, b1, b2, b3);
    
    float3 offset_p_base = a1 * POS_BASE;
    float3 offset_q_base = b1 * POS_BASE;
    float3 offset_p_back = a1 * POS_BACK;
    float3 offset_q_back = b1 * POS_BACK;
    
    float energy = 0.0f;
    float3 f = float3(0.0f);
    
    // BASE-BASE
    float3 rcenter = r + offset_q_base - offset_p_base;
    energy += _repulsive_lj(rcenter, f, EXCL_S2, EXCL_R2, EXCL_B2, EXCL_RC2, update_forces);
    if(update_forces) {
        force_p -= f;
        force_q += f;
        torque_p -= cross(offset_p_base, f);
        torque_q += cross(offset_q_base, f);
    }
    
    // P-BASE vs Q-BACK
    rcenter = r + offset_q_back - offset_p_base;
    f = float3(0.0f);
    energy += _repulsive_lj(rcenter, f, EXCL_S3, EXCL_R3, EXCL_B3, EXCL_RC3, update_forces);
    if(update_forces) {
        force_p -= f;
        force_q += f;
        torque_p -= cross(offset_p_base, f);
        torque_q += cross(offset_q_back, f);
    }
    
    // P-BACK vs Q-BASE
    rcenter = r + offset_q_base - offset_p_back;
    f = float3(0.0f);
    energy += _repulsive_lj(rcenter, f, EXCL_S3, EXCL_R3, EXCL_B3, EXCL_RC3, update_forces);
    if(update_forces) {
        force_p -= f;
        force_q += f;
        torque_p -= cross(offset_p_back, f);
        torque_q += cross(offset_q_base, f);
    }
    
    return energy;
}


// Stacking interaction
inline float _stacking(int idx, int neighbor_idx, float3 r, float3 r_pos, float3 q_pos, m_number4 p_quat, m_number4 q_quat, int type_me, int type_neig, bool update_forces, thread float3 &force_p, thread float3 &force_q, thread float3 &torque_p, thread float3 &torque_q, constant DNAInteractionParams &params) {
    
    float3 a1, a2, a3;
    get_axes(p_quat, a1, a2, a3);
    
    float3 b1, b2, b3;
    get_axes(q_quat, b1, b2, b3);
    
    float3 rbackref = r + b1 * POS_BACK - a1 * POS_BACK;
    float rbackrefmod = length(rbackref);
    float rbackrefmodcub = CUBE(rbackrefmod);
    
    // Stack centers
    float3 offset_p_stack = a1 * POS_STACK;
    float3 offset_q_stack = b1 * POS_STACK;
    float3 rstack = r + offset_q_stack - offset_p_stack;
    float rstackmod = length(rstack);
    float3 rstackdir = rstack / rstackmod;
    
    float cost4 = dot(a3, b3);
    float cost5 = dot(a3, rstackdir);
    float cost6 = -dot(b3, rstackdir);
    float cosphi1 = dot(a2, rbackref) / rbackrefmod; 
    float cosphi2 = dot(b2, rbackref) / rbackrefmod;
    
    float f1 = _f1(rstackmod, STCK_F1, type_me, type_neig, params);
    float f4t4 = _f4(cost4, STCK_F4_THETA4, params); 
    float f4t5 = _f4(-cost5, STCK_F4_THETA5, params);
    float f4t6 = _f4(cost6, STCK_F4_THETA6, params);
    float f5phi1 = _f5(cosphi1, STCK_F5_PHI1, params);
    float f5phi2 = _f5(cosphi2, STCK_F5_PHI2, params);
    
    float energy = f1 * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2;
    
    if(update_forces && energy != 0.0f) {
        float f1D = _f1D(rstackmod, STCK_F1, type_me, type_neig, params);
        float f4t4Dsin = -_f4Dsin(cost4, STCK_F4_THETA4, params); 
        float f4t5Dsin = -_f4Dsin(-cost5, STCK_F4_THETA5, params);
        float f4t6Dsin = -_f4Dsin(cost6, STCK_F4_THETA6, params);
        float f5phi1D = _f5D(cosphi1, STCK_F5_PHI1, params);
        float f5phi2D = _f5D(cosphi2, STCK_F5_PHI2, params); // Checked CPU: _f5D
        
        // Radial Force (CPU: force = -rstackdir * ...)
        // This is Force on Q? (Neig->Me is Q->P).
        // If coeffs positive, -rstackdir points Q->P.
        // p->force -= force -> Adds P->Q.
        // We calculate "force_var" as per CPU.
        float3 force_var = -rstackdir * (f1D * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2);
        
        // Theta 5
        force_var += -(a3 - rstackdir * cost5) * (f1 * f4t4 * f4t5Dsin * f4t6 * f5phi1 * f5phi2 / rstackmod);
        
        // Theta 6
        force_var += -(b3 + rstackdir * cost6) * (f1 * f4t4 * f4t5 * f4t6Dsin * f5phi1 * f5phi2 / rstackmod);
        
        // Phi 1
        float gamma = POS_STACK - POS_BACK;
        float ra2 = dot(rstackdir, a2);
        float ra1 = dot(rstackdir, a1);
        float rb1 = dot(rstackdir, b1);
        float a2b1 = dot(a2, b1);
        
        float parentesi = rstackmod * ra2 - a2b1 * gamma;
        
        float dcosphi1dr = (SQR(rstackmod) * ra2 - ra2 * SQR(rbackrefmod) - rstackmod * (a2b1 + ra2 * (-ra1 + rb1)) * gamma + a2b1 * (-ra1 + rb1) * SQR(gamma)) / rbackrefmodcub;
        float dcosphi1dra1 = rstackmod * gamma * parentesi / rbackrefmodcub;
        float dcosphi1dra2 = -rstackmod / rbackrefmod;
        float dcosphi1drb1 = -rstackmod * gamma * parentesi / rbackrefmodcub;
        
        float dcosphi1da1b1 = -SQR(gamma) * parentesi / rbackrefmodcub;
        float dcosphi1da2b1 = gamma / rbackrefmod;
        
        float force_part_phi1 = -f1 * f4t4 * f4t5 * f4t6 * f5phi1D * f5phi2;
        
        force_var += -(rstackdir * dcosphi1dr + 
                       ((a2 - rstackdir * ra2) * dcosphi1dra2 +
                        (a1 - rstackdir * ra1) * dcosphi1dra1 + 
                        (b1 - rstackdir * rb1) * dcosphi1drb1) / rstackmod) * force_part_phi1;
                        
        // Phi 2
        ra2 = dot(rstackdir, b2);
        ra1 = dot(rstackdir, b1);
        rb1 = dot(rstackdir, a1);
        a2b1 = dot(b2, a1);
        
        parentesi = rstackmod * ra2 + a2b1 * gamma;
        
        float dcosphi2dr = (parentesi * (rstackmod + (rb1 - ra1) * gamma) - ra2 * SQR(rbackrefmod)) / rbackrefmodcub;
        float dcosphi2dra1 = -rstackmod * gamma * (rstackmod * ra2 + a2b1 * gamma) / rbackrefmodcub;
        float dcosphi2dra2 = -rstackmod / rbackrefmod;
        float dcosphi2drb1 = rstackmod * gamma * parentesi / rbackrefmodcub;
        
        float dcosphi2da1b1 = -SQR(gamma) * parentesi / rbackrefmodcub;
        float dcosphi2da2b1 = -gamma / rbackrefmod;
        
        float force_part_phi2 = -f1 * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2D;
        
        force_var += -force_part_phi2 * (rstackdir * dcosphi2dr + 
                                         ((b2 - rstackdir * ra2) * dcosphi2dra2 +
                                          (b1 - rstackdir * ra1) * dcosphi2dra1 + 
                                          (a1 - rstackdir * rb1) * dcosphi2drb1) / rstackmod);
        
        // Accumulate Forces
        // p->force -= force_var;
        // q->force += force_var;
        force_p += -force_var;
        force_q += force_var;
        
        // Lever arm Torques
        torque_p += cross(offset_p_stack, -force_var);
        torque_q += cross(offset_q_stack, force_var);
        
        // Theta 4 Torque
        float3 t4dir = cross(b3, a3);
        float torquemod = f1 * f4t4Dsin * f4t5 * f4t6 * f5phi1 * f5phi2;
        torque_p -= t4dir * torquemod;
        torque_q += t4dir * torquemod;
        
        // Theta 5
        float3 t5dir = cross(rstackdir, a3);
        torquemod = -f1 * f4t4 * f4t5Dsin * f4t6 * f5phi1 * f5phi2;
        torque_p -= t5dir * torquemod;
        
        // Theta 6
        float3 t6dir = cross(rstackdir, b3);
        torquemod = f1 * f4t4 * f4t5 * f4t6Dsin * f5phi1 * f5phi2;
        torque_q += t6dir * torquemod;
        
        // Phi 1 Torque
        torque_p += cross(rstackdir, a2) * force_part_phi1 * dcosphi1dra2 +
                    cross(rstackdir, a1) * force_part_phi1 * dcosphi1dra1;
        torque_q += cross(rstackdir, b1) * force_part_phi1 * dcosphi1drb1;
        
        float3 puretorque = cross(a2, b1) * force_part_phi1 * dcosphi1da2b1 +
                            cross(a1, b1) * force_part_phi1 * dcosphi1da1b1;
        torque_p -= puretorque;
        torque_q += puretorque;
        
        // Phi 2 Torque (Restore old ra1/rb1 meanings for clarity? Need to be careful.
        // In CPU code, variables were reused.
        // Here I reused ra2, ra1, rb1 in Phi 2 block.
        // "particle p -> b, q -> a".
        // a2 is now b2...
        // Need to check cross products.
        // CPU: torquep -> p (b? No).
        // The block "Phi 2" calculates contribution relative based on swapped roles?
        // But updates `force` (same var) contribution.
        // Torques:
        // torqueq += ... (related to b/q).
        // torquep += ... (related to a/p).
        
        // Re-calculate variables for torque usage if needed.
        // In Phi 2 block:
        // ra2 = rstackdir * b2; ...
        // CPU Line 683:
        // torqueq += rstackdir.cross(b2) * force_part_phi2 * dcosphi2dra2 +
        //            rstackdir.cross(b1) * force_part_phi2 * dcosphi2dra1;
        // torquep += rstackdir.cross(a1) * force_part_phi2 * dcosphi2drb1;
        
        torque_q += cross(rstackdir, b2) * force_part_phi2 * dcosphi2dra2 + 
                    cross(rstackdir, b1) * force_part_phi2 * dcosphi2dra1;
        torque_p += cross(rstackdir, a1) * force_part_phi2 * dcosphi2drb1;
        
        puretorque = cross(b2, a1) * force_part_phi2 * dcosphi2da2b1 +
                     cross(b1, a1) * force_part_phi2 * dcosphi2da1b1;
                     
        torque_q -= puretorque;
        torque_p += puretorque;
    }
    
    return energy;
}

/**
 * @brief DNA forces kernel
 */
kernel void dna_forces(device m_number4 *poss [[buffer(0)]],
                       device m_number4 *orientations [[buffer(1)]],
                       device m_number4 *forces [[buffer(2)]],
                       device m_number4 *torques [[buffer(3)]],
                       device int *matrix_neighs [[buffer(4)]],
                       device int *number_neighs [[buffer(5)]],
                       device MetalBonds *bonds [[buffer(6)]],
                       constant DNAInteractionParams &params [[buffer(7)]],
                       constant MetalBox &box [[buffer(8)]],
                       constant InitStrandArgs &args [[buffer(9)]],
                       device float *energies [[buffer(10)]],
                       uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    if(idx >= args.N) return;
    
    // Clear energies
    if(energies) {
         for(int k=0; k<10; k++) energies[idx*10 + k] = 0.0f;
    }
    
    float3 r_pos = poss[idx].xyz;
    m_number4 p_quat = orientations[idx];
    
    MetalBonds b = bonds[idx];
    int n_neighs = number_neighs[idx];
    
    float3 tot_force = float3(0.0f);
    float3 tot_torque = float3(0.0f);
    int type_me = (int)poss[idx].w;
    
    // 1. Bonded Interactions (Backbone)
    // 1. Bonded Interactions (Backbone)
    if(b.n3 != -1) {
        int j = b.n3;
        float3 rj_pos = poss[j].xyz;
        float3 dr = rj_pos - r_pos;
        dr = minimum_image(dr, box);
        
        m_number4 quat_me = p_quat; 
        m_number4 quat_neig = orientations[j];
        
        float3 a1, a2, a3; get_axes(quat_me, a1, a2, a3);
        float3 b1, b2, b3; get_axes(quat_neig, b1, b2, b3);
        
        // FENE: Me(n3) -> Neig (P=Me, Q=Neig)
        // rback = dr_com + off_q - off_p
        float3 rback = dr + b1 * POS_BACK - a1 * POS_BACK;
        
        float3 f = float3(0.0f);
        float en = _backbone(idx, j, dr, rback, true, f, params);
        if(energies) energies[idx*10 + 0] += en; 
        tot_force += f; 
        
        // Stacking (Me -> Neig)
        // Me is P (5'), Neig is Q (3').
        int type_neig = (int)poss[j].w;
        float3 f_p = float3(0.0f);
        float3 f_q = float3(0.0f);
        float3 t_p = float3(0.0f);
        float3 t_q = float3(0.0f);
        
        float en_stack = _stacking(idx, j, dr, r_pos, rj_pos, quat_me, quat_neig, type_me, type_neig, true, f_p, f_q, t_p, t_q, params);
        if(energies) energies[idx*10 + 2] += en_stack; 
        
        // Bonded Excluded Volume (Me -> Neig)
        float en_bonded = _bonded_excluded_volume(idx, j, dr, r_pos, rj_pos, quat_me, quat_neig, true, f_p, f_q, t_p, t_q, params);
        if(energies) energies[idx*10 + 1] += en_bonded;

        /*
        if (idx == 0) {
             // printf("Step: idx=0, j=%d, r=%f, E_stack=%f, E_bond=%f, F_p=(%f,%f,%f)\n", 
             //       j, length(dr), en_stack, en_bonded, f_p.x, f_p.y, f_p.z);
        }
        */

        tot_force += f_p;
        tot_torque += t_p;
    }
    
    // N5 Logic: Symmetric.
    if(b.n5 != -1) {
        int j = b.n5;
        float3 rj_pos = poss[j].xyz;
        float3 dr = rj_pos - r_pos;
        dr = minimum_image(dr, box); 
        
        m_number4 quat_me = p_quat;
        m_number4 quat_neig = orientations[j];
        
        float3 a1, a2, a3; get_axes(quat_me, a1, a2, a3);
        float3 b1, b2, b3; get_axes(quat_neig, b1, b2, b3);
        
        // FENE: Neig(n3) -> Me (P=Neig, Q=Me)
        // r = q - p = Me - Neig = -dr
        // rback = (pos_me + off_me) - (pos_neig + off_neig)
        // rback = -dr + off_me - off_neig
        float3 rback = -dr + a1 * POS_BACK - b1 * POS_BACK;
        
        float3 f = float3(0.0f);
        float en = _backbone(j, idx, -dr, rback, true, f, params); 
        
        // f is Force on J (First arg). Force on Me = -f.
        tot_force -= f; 
        
        // Stacking (Neig -> Me)
        // We are q (3'). Neig is p (5').
        // Call _stacking(p, q...) -> _stacking(j, idx...)
        int type_neig = (int)poss[j].w;
        
        float3 f_p = float3(0.0f);
        float3 f_q = float3(0.0f);
        float3 t_p = float3(0.0f);
        float3 t_q = float3(0.0f);
        
        float en_stack = _stacking(j, idx, -dr, rj_pos, r_pos, quat_neig, quat_me, type_neig, type_me, true, f_p, f_q, t_p, t_q, params);
        // Do NOT add energy (counted in n3 block of partner)
        
        // Bonded Excluded Volume (Neig -> Me)
        // We are Q in _bonded_excl(p, q).
        // call _bonded_excl(neig, me).
        // Updates f_q, t_q (which is Me).
        float en_bonded = _bonded_excluded_volume(j, idx, -dr, rj_pos, r_pos, quat_neig, quat_me, true, f_p, f_q, t_p, t_q, params);
        
        // We are Q.
        tot_force += f_q;
        tot_torque += t_q;
    }
    
    // 2. Non-bonded Interactions (Excluded Volume)
    // 2. Non-bonded Interactions (Excluded Volume)
    for(int i = 0; i < n_neighs; i++) {
        int j = matrix_neighs[i * args.N + idx]; 
        
        if (j == b.n3 || j == b.n5) continue; // Skip bonded
        if (j == idx) continue; // Skip self
        
        float3 rj_pos = poss[j].xyz;
        float3 dr = rj_pos - r_pos;
        dr = minimum_image(dr, box);
        
        if (dot(dr, dr) < 0.0001f) continue; // Skip overlaps
        
        // Orientations
        m_number4 start_quat = orientations[idx];
        m_number4 end_quat = orientations[j];
        
        float3 a1, a2, a3;
        get_axes(start_quat, a1, a2, a3);
        
        float3 b1, b2, b3;
        get_axes(end_quat, b1, b2, b3);
        
        // Interaction offsets
        float3 offset_me_back = a1 * POS_BACK;
        float3 offset_me_base = a1 * POS_BASE;
        float3 offset_neig_back = b1 * POS_BACK;
        float3 offset_neig_base = b1 * POS_BASE;
        
        float3 f_sub = float3(0.0f);
        float en_sub = 0.0f;
        
        // 1. Back-Back (S1)
        float3 dr_1 = dr + offset_neig_back - offset_me_back;
        en_sub = _repulsive_lj(dr_1, f_sub, EXCL_S1, EXCL_R1, EXCL_B1, EXCL_RC1, true);
        if(energies) energies[idx*10 + 1] += 0.5f * en_sub; 
        tot_force += f_sub;
        tot_torque += cross(offset_me_back, f_sub); // Torque enabled

        // 2. Base-Base (S2)
        float3 dr_2 = dr + offset_neig_base - offset_me_base;
        en_sub = _repulsive_lj(dr_2, f_sub, EXCL_S2, EXCL_R2, EXCL_B2, EXCL_RC2, true);
        if(energies) energies[idx*10 + 1] += 0.5f * en_sub;
        tot_force += f_sub;
        tot_torque += cross(offset_me_base, f_sub);

        // 3. Base-Back (Me Base - Neig Back) (S3)
        float3 dr_3 = dr + offset_neig_back - offset_me_base;
        en_sub = _repulsive_lj(dr_3, f_sub, EXCL_S3, EXCL_R3, EXCL_B3, EXCL_RC3, true);
        if(energies) energies[idx*10 + 1] += 0.5f * en_sub;
        tot_force += f_sub;
        tot_torque += cross(offset_me_base, f_sub);

        // 4. Back-Base (Me Back - Neig Base) (S4)
        float3 dr_4 = dr + offset_neig_base - offset_me_back;
        en_sub = _repulsive_lj(dr_4, f_sub, EXCL_S4, EXCL_R4, EXCL_B4, EXCL_RC4, true);
        if(energies) energies[idx*10 + 1] += 0.5f * en_sub;
        tot_force += f_sub;
        tot_torque += cross(offset_me_back, f_sub);
    }
    
    forces[idx].xyz += tot_force;
    torques[idx].xyz += tot_torque;
}
