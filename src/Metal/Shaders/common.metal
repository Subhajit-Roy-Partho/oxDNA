/**
 * @file    common.metal
 * @brief   Common Metal shader functions and data structures
 *
 * Metal shader equivalent of common CUDA kernels
 */

#include <metal_stdlib>
using namespace metal;

// Precision selection
#ifdef METAL_DOUBLE_PRECISION
typedef double m_number;
typedef double3 m_number3;
typedef double4 m_number4;
#else
typedef float m_number;
typedef float3 m_number3;
typedef float4 m_number4;
#endif

/**
 * @brief Simulation box structure
 */
struct MetalBox {
    m_number box_sides[3];
    m_number inv_sides[3];
};

/**
 * @brief Bond information
 */
struct MetalBonds {
    int n3;
    int n5;
};

/**
 * @brief Stress tensor
 */
struct MetalStressTensor {
    m_number e[6];
};

/**
 * @brief Minimum image convention for periodic boundary conditions
 */
inline m_number3 minimum_image(m_number3 r, constant MetalBox &box) {
    m_number3 result = r;

    for(int i = 0; i < 3; i++) {
        result[i] -= box.box_sides[i] * rint(result[i] * box.inv_sides[i]);
    }

    return result;
}

/**
 * @brief Calculate distance squared between two positions with PBC
 */
inline m_number distance_sqr(m_number3 r1, m_number3 r2, constant MetalBox &box) {
    m_number3 dr = r1 - r2;
    dr = minimum_image(dr, box);
    return dot(dr, dr);
}

/**
 * @brief Safe arc cosine with clamping
 */
inline m_number safe_acos(m_number x) {
    if(x >= 1.0) return 0.0;
    if(x <= -1.0) return M_PI_F;
    return acos(x);
}

/**
 * @brief Cross product
 */
inline m_number3 cross_product(m_number3 a, m_number3 b) {
    return cross(a, b);
}

/**
 * @brief Normalize vector
 */
inline m_number3 normalize_vector(m_number3 v) {
    return normalize(v);
}

/**
 * @brief Atomic add for floating point (Metal doesn't have native atomic float add)
 */
inline void atomic_add_float(device atomic<float> *addr, float value) {
    // Use compare-and-swap loop
    float old_val = atomic_load_explicit(addr, memory_order_relaxed);
    float new_val;
    do {
        new_val = old_val + value;
    } while (!atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed));
}

/**
 * @brief Quaternion rotation
 */
inline m_number3 quaternion_rotate(m_number4 q, m_number3 v) {
    // q = (x, y, z, w) where w is the scalar part
    m_number3 qv = m_number3(q.x, q.y, q.z);
    m_number3 t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

/**
 * @brief Parallel reduction for sum (threadgroup memory)
 */
template<typename T>
inline T threadgroup_reduce_add(T value, threadgroup T *shared_data, uint tid, uint threads) {
    shared_data[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within threadgroup
    for(uint s = threads / 2; s > 0; s >>= 1) {
        if(tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return shared_data[0];
}
