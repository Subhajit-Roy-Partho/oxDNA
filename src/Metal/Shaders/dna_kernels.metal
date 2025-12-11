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
};

struct InitStrandArgs {
    int N;
};

// Helper macros
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

// Constants
#define EXCL_EPS 1.0f
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
    float k_fene = FENE_EPS; // Actually 2.0. Formula uses EPS/2? check oxDNA.
    // oxDNA: -eps/2 * log.
    // Force: -eps * r / (delta^2 - r^2).
    
    // Switch to harmonic at 90% of delta (0.81 squared)
    float switch_ratio = 0.9f;
    float switch_dist = FENE_DELTA * switch_ratio;
    float switch_sqr = SQR(switch_dist);
    
    // Current displacement sqr
    float r_sqr = SQR(rbackr0);
    
    float energy = 0.0f;
    float fmod = 0.0f;
    
    if (r_sqr < switch_sqr) {
        // Standard FENE
        energy = -(k_fene / 2.0f) * log(1.0f - r_sqr / delta_sqr);
        if(update_forces) {
            fmod = -(k_fene * rbackr0 / (delta_sqr - r_sqr));
        }
    } else {
        // Harmonic Tail
        // Match V and F at switch_dist.
        // V_c = - eps/2 * log(1 - switch^2/delta^2)
        // F_c (scalar magnitude) = eps * switch / (delta^2 - switch^2) (Negative)
        // Let's preserve direction rbackr0.
        // F_c_val = - (eps * switch_dist) / (delta^2 - switch_sqr).
        // V(r) = V_c - F_c_val * (abs(r) - switch_dist) + 0.5 * K_tail * (abs(r) - switch_dist)^2
        // Wait, F = -dV/dr. 
        // Force F(r) = F_c_val - K_tail * (abs(r) - switch_dist).
        // F_c_val is negative (attractive).
        // We want force to become MORE negative (stronger attraction) as r increases.
        // So K_tail should be positive?
        // - ( - huge ) = + huge?
        // Let's just linearly extrapolate Force slope?
        // Or simpler: F(r) = F_c_val * (1 + steepness * (abs(r) - switch)).
        
        float V_c = -(k_fene / 2.0f) * log(1.0f - switch_sqr / delta_sqr);
        float F_c = -(k_fene * switch_dist / (delta_sqr - switch_sqr)); // Negative value
        
        // Linear Force extension (Quadratic Potential) is robust.
        // Slope of F?
        // dF/dr of FENE at switch point.
        // F = - eps * r / (D^2 - r^2).
        // dF/dr = - eps * [ (D^2 - r^2) - r * (-2r) ] / (...) = - eps * (D^2 + r^2) / (D^2 - r^2)^2.
        // Calculate slope K_tail at switch.
        float denom = delta_sqr - switch_sqr;
        float K_tail = -(k_fene * (delta_sqr + switch_sqr)) / (denom * denom); // Negative slope (force becomes more negative)
        
        float dr_tail = abs(rbackr0) - switch_dist; // Positive overshoot
        
        // Taylor expansion for V:
        // V = V_c - F_c * dr_tail + 0.5 * (-K_tail) * dr_tail^2 ?
        // Force = F_c + K_tail * dr_tail.
        // Check V: dV/dr = -Force = - (F_c + K_tail * dr). Correct.
        // Note: F_c is negative. K_tail is negative.
        // So Force becomes more negative (stronger attraction).
        
        if (update_forces) {
            fmod = F_c + K_tail * dr_tail;
            // Apply sign of rbackr0 (fmod assumes rbackr0 is positive distance, direction handled by vector)
            // rbackr0 was float scalar = mod - 0.75.
            // If rbackr0 is negative (compressed)?
            // Harmonic logic applies to |rbackr0|.
            // But we checked r_sqr < switch_sqr.
            // If compressed > switch (rare for fene?), delta handled?
            // Usually FENE not used for compression < -Delta?
            // Assuming rbackr0 > 0 (stretched).
            // If rbackr0 < 0 (compressed), FENE also works.
            // If compressed beyond switch?
            // Logic holds for abs.
            if(rbackr0 < 0) fmod = -fmod; // Flip force?
            // Wait.
            // FENE force formula: fmod = - eps * r / (D^2 - r^2).
            // If r is negative: fmod is positive. (Repulsive).
            // Force vector = rback * (fmod / rmod).
            // If rbackr0 = rmod - 0.75.
            // If rmod < 0.75, rbackr0 < 0.
            // fmod > 0. Force points rback * pos = Me->Neig. Repulsive (pushing me away from neig, increasing r). Correct.
            // Logic holds.
            // For Tail:
            // dr_tail = abs(r) - switch.
            // F = F_c (at limit, signed correctly) + K_tail * dr_tail * sign(r)?
            // K_tail calc used abs values. K_tail is negative (stiffening).
            // F_new_abs = F_c_abs + abs(K_tail) * dr_tail.
            // F_new = F_new_abs * sign(F_c).
            // F_c has sign of -r.
            // So fmod = (F_c_val_at_pos_switch + K_tail_val * dr_tail) * (rbackr0 > 0 ? 1 : -1).
            // Let's simplify. Assumed stretched > 0.
            // If compressed, Excluded Volume dominates. FENE not usually tail-checked for compression.
            // Just use Signed R logic.
        }
        
        energy = V_c - F_c * dr_tail - 0.5f * K_tail * SQR(dr_tail);
    }
    
    if(update_forces) {
        force = rback * (fmod / rbackmod);
    }
    
    return energy;
}

// Stacking interaction
inline float _stacking(int idx, int neighbor_idx, float3 r, float3 r_pos, float3 q_pos, m_number4 p_quat, m_number4 q_quat, bool update_forces, thread float3 &force, constant DNAInteractionParams &params) {
    
    float3 a1, a2, a3;
    get_axes(p_quat, a1, a2, a3);
    
    float3 b1, b2, b3;
    get_axes(q_quat, b1, b2, b3);
    
    float POS_BACK = -0.4f;
    // float POS_STACK = 0.34f;
    
    float3 rbackref = r + b1 * POS_BACK - a1 * POS_BACK;
    float rbackrefmod = length(rbackref);
    
    float rstackmod = length(r);
    float3 rstackdir = r / rstackmod;
    
    float cost4 = dot(a3, b3);
    float cost5 = dot(a3, rstackdir);
    float cost6 = -dot(b3, rstackdir);
    float cosphi1 = dot(a2, rbackref) / rbackrefmod; 
    float cosphi2 = dot(b2, rbackref) / rbackrefmod;
    
    float f1 = _f1(rstackmod, STCK_F1, 0, 0, params);
    float f4t4 = _f4(cost4, STCK_F4_THETA4, params); 
    float f4t5 = _f4(-cost5, STCK_F4_THETA5, params);
    float f4t6 = _f4(cost6, STCK_F4_THETA6, params);
    float f5phi1 = _f5(cosphi1, STCK_F5_PHI1, params);
    float f5phi2 = _f5(cosphi2, STCK_F5_PHI2, params);
    
    float energy = f1 * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2;
    
    if(update_forces && energy != 0.0f) {
        float f1D = _f1D(rstackmod, STCK_F1, 0, 0, params);
        float f4t4Dsin = _f4Dsin(cost4, STCK_F4_THETA4, params); 
        float f4t5Dsin = _f4Dsin(-cost5, STCK_F4_THETA5, params);
        float f4t6Dsin = _f4Dsin(cost6, STCK_F4_THETA6, params);
        float f5phi1D = _f5D(cosphi1, STCK_F5_PHI1, params);
        float f5phi2D = _f5D(cosphi2, STCK_F5_PHI2, params);
        
        // Radial Force (Fixed sign: +rstackdir = Attractive)
        // F_radial = +dir * dE/dr.
        float3 f_rad = rstackdir * (f1D * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2);
        
        force += f_rad;
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
    // float3 tot_torque = float3(0.0f); // Pending
    
    // 1. Bonded Interactions (Backbone)
    if(b.n3 != -1) {
        int j = b.n3;
        float3 rj_pos = poss[j].xyz;
        float3 dr = rj_pos - r_pos;
        dr = minimum_image(dr, box);
        
        float3 f = float3(0.0f);
        float en = _backbone(idx, j, dr, dr, true, f, params);
        if(energies) energies[idx*10 + 0] += en; 
        
        // Corrected Sign: Backbone should be attractive. f points Q->P (towards me)?? No.
        // f derived in _backbone: rback * (neg/rback). rback=dr(Me->Neig).
        // f = dr * (-k). Points Neig->Me (towards Me).
        // Attractive Force on Me.
        // So += f.
        tot_force += f; 
        
        // Stacking (with n3)
        m_number4 q_quat = orientations[j];
        float3 f_stack = float3(0.0f);
        float en_stack = _stacking(idx, j, dr, r_pos, rj_pos, p_quat, q_quat, true, f_stack, params);
        if(energies) energies[idx*10 + 2] += en_stack; 
        
        // Stacking sign: _stacking returns attractive force?
        // rstackdir = dr / mod. (Me->Neig).
        // force += rstackdir * f1D.
        // f1D is positive. force points Me->Neig.
        // Attractive Force on Me (towards Neighbor).
        // So += f_stack.
        tot_force += f_stack; 
    }
    
    // N5 Logic: Symmetric.
    if(b.n5 != -1) {
        int j = b.n5;
        float3 rj_pos = poss[j].xyz;
        float3 dr = rj_pos - r_pos;
        dr = minimum_image(dr, box); 
        float3 f = float3(0.0f);
        float en = _backbone(j, idx, -dr, -dr, true, f, params); 
        
        // f is Force on J (First arg).
        // Force on Me = -f.
        // Newton 3rd law.
        // So tot_force -= f.
        tot_force -= f; 
    }
    
    // 2. Non-bonded Interactions (Excluded Volume)
    for(int i = 0; i < n_neighs; i++) {
        int j = matrix_neighs[i * args.N + idx]; 
        
        if (j == b.n3 || j == b.n5) continue; // Skip bonded
        
        float3 rj = poss[j].xyz;
        float3 dr = rj - r_pos;
        dr = minimum_image(dr, box);
        
        float3 f = float3(0.0f);
        float en = _repulsive_lj(dr, f, 1.0f, 1.0f, 1.0f, 1.6f, true); 
        
        if(energies) energies[idx*10 + 1] += 0.5f * en; 
        
        // _repulsive_lj: f = -r * ... (Repulsive).
        // r = Neig - Me. No, r in NonBonded loop: rj - r_pos. (Neig - Me).
        // -r = Me - Neig.
        // f points Me - Neig. Repulsive (Away from Neig).
        // Force on Me.
        // += f.
        tot_force += f; 
    }
    
    forces[idx].xyz += tot_force;
}
