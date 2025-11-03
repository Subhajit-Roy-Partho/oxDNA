#include "dna2_mathematical_functions.h"
#include <cmath>
#include <stdexcept>

namespace dna2 {

// MathematicalFunctions implementation
torch::Tensor MathematicalFunctions::f1(const torch::Tensor& r, FunctionType type, 
                                       const torch::Tensor& n3_types, 
                                       const torch::Tensor& n5_types,
                                       const DNA2Parameters& params) {
    // Get sequence-dependent parameters
    torch::Tensor eps;
    torch::Tensor shift;
    
    if (type == FunctionType::F1_HYDR) {
        eps = params.hydr_eps_matrix.index({n5_types, n3_types});
        shift = torch::zeros_like(eps);  // Simplified - should use actual shift values
    } else if (type == FunctionType::F1_STCK) {
        eps = params.stck_eps_matrix.index({n5_types, n3_types});
        shift = torch::zeros_like(eps);  // Simplified - should use actual shift values
    } else {
        throw DNA2EnergyException("Unknown function type for f1");
    }
    
    // Get parameters based on type
    float rlow, rhigh, rchigh, rc_low, a, blow, bhigh;
    
    if (type == FunctionType::F1_HYDR) {
        rlow = 0.0f;  // Simplified values
        rhigh = params.hydr_rc;
        rchigh = params.hydr_rc + 0.1f;
        rc_low = params.hydr_r0;
        a = params.hydr_a;
        blow = 2.0f;
        bhigh = 2.0f;
    } else {  // F1_STCK
        rlow = 0.0f;  // Simplified values
        rhigh = params.stck_rc;
        rchigh = params.stck_rc + 0.1f;
        rc_low = params.stck_r0;
        a = params.stck_a;
        blow = 2.0f;
        bhigh = 2.0f;
    }
    
    // Create masks for different regions
    auto mask_low = r < rlow;
    auto mask_mid = (r >= rlow) & (r <= rhigh);
    auto mask_high = (r > rhigh) & (r < rchigh);
    
    // Compute values for each region
    auto val_low = blow * torch::square(r - rc_low);
    auto exp_term = torch::exp(-(r - rc_low) * a);
    auto val_mid = torch::square(1.0f - exp_term);
    auto val_high = bhigh * torch::square(r - rchigh);
    
    // Combine results with proper masking
    auto result = torch::zeros_like(r);
    result = torch::where(mask_low, eps * val_low, result);
    result = torch::where(mask_mid, eps * val_mid - shift, result);
    result = torch::where(mask_high, eps * val_high, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f1_derivative(const torch::Tensor& r, FunctionType type,
                                                  const torch::Tensor& n3_types,
                                                  const torch::Tensor& n5_types,
                                                  const DNA2Parameters& params) {
    // Get sequence-dependent parameters
    torch::Tensor eps;
    
    if (type == FunctionType::F1_HYDR) {
        eps = params.hydr_eps_matrix.index({n5_types, n3_types});
    } else if (type == FunctionType::F1_STCK) {
        eps = params.stck_eps_matrix.index({n5_types, n3_types});
    } else {
        throw DNA2EnergyException("Unknown function type for f1 derivative");
    }
    
    // Get parameters based on type
    float rlow, rhigh, rchigh, rc_low, a, blow, bhigh;
    
    if (type == FunctionType::F1_HYDR) {
        rlow = 0.0f;
        rhigh = params.hydr_rc;
        rchigh = params.hydr_rc + 0.1f;
        rc_low = params.hydr_r0;
        a = params.hydr_a;
        blow = 2.0f;
        bhigh = 2.0f;
    } else {  // F1_STCK
        rlow = 0.0f;
        rhigh = params.stck_rc;
        rchigh = params.stck_rc + 0.1f;
        rc_low = params.stck_r0;
        a = params.stck_a;
        blow = 2.0f;
        bhigh = 2.0f;
    }
    
    // Create masks
    auto mask_low = r < rlow;
    auto mask_mid = (r >= rlow) & (r <= rhigh);
    auto mask_high = (r > rhigh) & (r < rchigh);
    
    // Compute derivatives for each region
    auto val_low = 2.0f * blow * (r - rc_low);
    auto exp_term = torch::exp(-(r - rc_low) * a);
    auto val_mid = 2.0f * (1.0f - exp_term) * exp_term * a;
    auto val_high = 2.0f * bhigh * (r - rchigh);
    
    // Combine results
    auto result = torch::zeros_like(r);
    result = torch::where(mask_low, eps * val_low, result);
    result = torch::where(mask_mid, eps * val_mid, result);
    result = torch::where(mask_high, eps * val_high, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f2(const torch::Tensor& r, FunctionType type,
                                       const DNA2Parameters& params) {
    float k, rlow, rhigh, rchigh, rc_low, blow, bhigh, r0, rc;
    
    if (type == FunctionType::F2_CRST) {
        k = params.crst_k;
        r0 = params.crst_r0;
        rc = params.crst_rc;
        rlow = 0.0f;
        rhigh = rc;
        rchigh = rc + 0.1f;
        rc_low = r0;
        blow = 2.0f;
        bhigh = 2.0f;
    } else if (type == FunctionType::F2_CXST) {
        k = params.cxst_k;
        r0 = params.cxst_r0;
        rc = params.cxst_rc;
        rlow = 0.0f;
        rhigh = rc;
        rchigh = rc + 0.1f;
        rc_low = r0;
        blow = 2.0f;
        bhigh = 2.0f;
    } else {
        throw DNA2EnergyException("Unknown function type for f2");
    }
    
    // Create masks
    auto mask_low = r < rlow;
    auto mask_mid = (r >= rlow) & (r <= rhigh);
    auto mask_high = (r > rhigh) & (r < rchigh);
    
    // Compute values
    auto val_low = k * blow * torch::square(r - rc_low);
    auto val_mid = 0.5f * k * (
        torch::square(r - r0) - 
        torch::square(rc - r0)
    );
    auto val_high = k * bhigh * torch::square(r - rchigh);
    
    // Combine results
    auto result = torch::zeros_like(r);
    result = torch::where(mask_low, val_low, result);
    result = torch::where(mask_mid, val_mid, result);
    result = torch::where(mask_high, val_high, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f2_derivative(const torch::Tensor& r, FunctionType type,
                                                  const DNA2Parameters& params) {
    float k, rlow, rhigh, rchigh, rc_low, blow, bhigh, r0, rc;
    
    if (type == FunctionType::F2_CRST) {
        k = params.crst_k;
        r0 = params.crst_r0;
        rc = params.crst_rc;
        rlow = 0.0f;
        rhigh = rc;
        rchigh = rc + 0.1f;
        rc_low = r0;
        blow = 2.0f;
        bhigh = 2.0f;
    } else if (type == FunctionType::F2_CXST) {
        k = params.cxst_k;
        r0 = params.cxst_r0;
        rc = params.cxst_rc;
        rlow = 0.0f;
        rhigh = rc;
        rchigh = rc + 0.1f;
        rc_low = r0;
        blow = 2.0f;
        bhigh = 2.0f;
    } else {
        throw DNA2EnergyException("Unknown function type for f2 derivative");
    }
    
    // Create masks
    auto mask_low = r < rlow;
    auto mask_mid = (r >= rlow) & (r <= rhigh);
    auto mask_high = (r > rhigh) & (r < rchigh);
    
    // Compute derivatives
    auto val_low = 2.0f * k * blow * (r - rc_low);
    auto val_mid = k * (r - r0);
    auto val_high = 2.0f * k * bhigh * (r - rchigh);
    
    // Combine results
    auto result = torch::zeros_like(r);
    result = torch::where(mask_low, val_low, result);
    result = torch::where(mask_mid, val_mid, result);
    result = torch::where(mask_high, val_high, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f4(const torch::Tensor& theta, FunctionType type,
                                       const DNA2Parameters& params) {
    float t0, tc, ts, a, b;
    
    // Get parameters based on function type
    switch (type) {
        case FunctionType::F4_HYDR_T1:
            t0 = 2.35619449f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T2:
            t0 = 2.35619449f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T3:
            t0 = 0.785398163f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T4:
            t0 = 0.0f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T7:
            t0 = 0.785398163f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T8:
            t0 = 0.0f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T1:
            t0 = 0.959931089f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T2:
            t0 = 0.959931089f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T3:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T4:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_CRST_T1:
            t0 = 0.523598776f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_CRST_T2:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_CXST_T1:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        default:
            throw DNA2EnergyException("Unknown function type for f4");
    }
    
    // Compute absolute deviation from t0
    auto dt = theta - t0;
    auto abs_dt = torch::abs(dt);
    
    // Create masks
    auto mask_active = abs_dt < tc;
    auto mask_smooth = (abs_dt >= ts) & (abs_dt < tc);
    auto mask_core = abs_dt <= ts;
    
    // Compute values
    auto val_smooth = b * torch::square(tc - abs_dt);
    auto val_core = 1.0f - a * torch::square(abs_dt);
    
    // Combine results
    auto result = torch::zeros_like(theta);
    result = torch::where(mask_smooth, val_smooth, result);
    result = torch::where(mask_core, val_core, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f4_derivative(const torch::Tensor& theta, FunctionType type,
                                                  const DNA2Parameters& params) {
    float t0, tc, ts, a, b;
    
    // Get parameters based on function type (same as f4)
    switch (type) {
        case FunctionType::F4_HYDR_T1:
            t0 = 2.35619449f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T2:
            t0 = 2.35619449f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T3:
            t0 = 0.785398163f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T4:
            t0 = 0.0f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T7:
            t0 = 0.785398163f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_HYDR_T8:
            t0 = 0.0f; tc = 0.785398163f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T1:
            t0 = 0.959931089f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T2:
            t0 = 0.959931089f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T3:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_STCK_T4:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_CRST_T1:
            t0 = 0.523598776f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_CRST_T2:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case FunctionType::F4_CXST_T1:
            t0 = 0.0f; tc = 0.523598776f; ts = 0.174532925f;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        default:
            throw DNA2EnergyException("Unknown function type for f4 derivative");
    }
    
    // Compute absolute deviation and sign
    auto dt = theta - t0;
    auto abs_dt = torch::abs(dt);
    auto sign = torch::sign(dt);
    
    // Create masks
    auto mask_active = abs_dt < tc;
    auto mask_smooth = (abs_dt >= ts) & (abs_dt < tc);
    auto mask_core = abs_dt <= ts;
    
    // Compute derivatives
    auto val_smooth = sign * 2.0f * b * (abs_dt - tc);
    auto val_core = -sign * 2.0f * a * abs_dt;
    
    // Combine results
    auto result = torch::zeros_like(theta);
    result = torch::where(mask_smooth, val_smooth, result);
    result = torch::where(mask_core, val_core, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f4_derivative_sin(const torch::Tensor& theta, FunctionType type,
                                                      const DNA2Parameters& params) {
    auto derivative = f4_derivative(theta, type, params);
    auto sin_theta = torch::sin(theta);
    
    // Handle numerical issues when sin(theta) is close to zero
    auto eps = 1e-8f;
    auto mask_small_sin = torch::abs(sin_theta) < eps;
    
    auto result = derivative / sin_theta;
    
    // For small sin(theta), use L'Hôpital's rule approximation
    float t0, a;
    switch (type) {
        case FunctionType::F4_HYDR_T1:
        case FunctionType::F4_HYDR_T2:
            t0 = 2.35619449f; a = 0.5f / (0.174532925f * 0.174532925f);
            break;
        case FunctionType::F4_HYDR_T3:
        case FunctionType::F4_HYDR_T7:
            t0 = 0.785398163f; a = 0.5f / (0.174532925f * 0.174532925f);
            break;
        case FunctionType::F4_HYDR_T4:
        case FunctionType::F4_HYDR_T8:
        case FunctionType::F4_STCK_T3:
        case FunctionType::F4_STCK_T4:
        case FunctionType::F4_CRST_T2:
        case FunctionType::F4_CXST_T1:
            t0 = 0.0f; a = 0.5f / (0.174532925f * 0.174532925f);
            break;
        case FunctionType::F4_STCK_T1:
        case FunctionType::F4_STCK_T2:
            t0 = 0.959931089f; a = 0.5f / (0.174532925f * 0.174532925f);
            break;
        case FunctionType::F4_CRST_T1:
            t0 = 0.523598776f; a = 0.5f / (0.174532925f * 0.174532925f);
            break;
        default:
            throw DNA2EnergyException("Unknown function type for f4 derivative sin");
    }
    
    auto approximation = -2.0f * a * (theta - t0);
    result = torch::where(mask_small_sin, approximation, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f5(const torch::Tensor& phi, FunctionType type,
                                       const DNA2Parameters& params) {
    float xc, xs, a, b;
    
    if (type == FunctionType::F5_HYDR_PHI || type == FunctionType::F5_STCK_PHI) {
        // Simplified parameters - should use actual values from oxDNA2
        xc = -0.5f;
        xs = 0.0f;
        a = 2.0f;
        b = 2.0f;
    } else {
        throw DNA2EnergyException("Unknown function type for f5");
    }
    
    // Create masks
    auto mask_inactive = phi <= xc;
    auto mask_smooth = (phi > xc) & (phi < xs);
    auto mask_core = (phi >= xs) & (phi < 0);
    auto mask_positive = phi >= 0;
    
    // Compute values
    auto val_smooth = b * torch::square(xc - phi);
    auto val_core = 1.0f - a * torch::square(phi);
    auto val_positive = 1.0f;
    
    // Combine results
    auto result = torch::zeros_like(phi);
    result = torch::where(mask_smooth, val_smooth, result);
    result = torch::where(mask_core, val_core, result);
    result = torch::where(mask_positive, val_positive, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::f5_derivative(const torch::Tensor& phi, FunctionType type,
                                                  const DNA2Parameters& params) {
    float xc, xs, a, b;
    
    if (type == FunctionType::F5_HYDR_PHI || type == FunctionType::F5_STCK_PHI) {
        // Simplified parameters - should use actual values from oxDNA2
        xc = -0.5f;
        xs = 0.0f;
        a = 2.0f;
        b = 2.0f;
    } else {
        throw DNA2EnergyException("Unknown function type for f5 derivative");
    }
    
    // Create masks
    auto mask_smooth = (phi > xc) & (phi < xs);
    auto mask_core = (phi >= xs) & (phi < 0);
    
    // Compute derivatives
    auto val_smooth = 2.0f * b * (phi - xc);
    auto val_core = -2.0f * a * phi;
    
    // Combine results
    auto result = torch::zeros_like(phi);
    result = torch::where(mask_smooth, val_smooth, result);
    result = torch::where(mask_core, val_core, result);
    
    return result;
}

torch::Tensor MathematicalFunctions::smooth_step(const torch::Tensor& x, float x0, float x1) {
    auto t = torch::clamp((x - x0) / (x1 - x0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

torch::Tensor MathematicalFunctions::smooth_step_derivative(const torch::Tensor& x, float x0, float x1) {
    auto mask = (x >= x0) & (x <= x1);
    auto derivative = 6.0f * t * (1.0f - t) / (x1 - x0);
    return torch::where(mask, derivative, torch::zeros_like(x));
}

// TensorOperations implementation
torch::Tensor TensorOperations::cross_product(const torch::Tensor& a, const torch::Tensor& b) {
    return torch::cross(a, b, -1);
}

torch::Tensor TensorOperations::dot_product(const torch::Tensor& a, const torch::Tensor& b) {
    return torch::sum(a * b, -1);
}

torch::Tensor TensorOperations::normalize(const torch::Tensor& v) {
    return safe_normalize(v);
}

torch::Tensor TensorOperations::norm(const torch::Tensor& v) {
    return torch::norm(v, 2, -1, true);
}

torch::Tensor TensorOperations::matrix_transpose(const torch::Tensor& m) {
    return m.transpose(-2, -1);
}

torch::Tensor TensorOperations::matrix_multiply(const torch::Tensor& a, const torch::Tensor& b) {
    return torch::matmul(a, b);
}

torch::Tensor TensorOperations::compute_angles(const torch::Tensor& v1, const torch::Tensor& v2) {
    auto cos_angles = dot_product(normalize(v1), normalize(v2));
    return safe_acos(cos_angles);
}

torch::Tensor TensorOperations::pairwise_distances(const torch::Tensor& positions) {
    auto diff = positions.unsqueeze(1) - positions.unsqueeze(0);
    return torch::norm(diff, 2, -1);
}

torch::Tensor TensorOperations::pairwise_vectors(const torch::Tensor& positions) {
    return positions.unsqueeze(1) - positions.unsqueeze(0);
}

torch::Tensor TensorOperations::safe_normalize(const torch::Tensor& v, float eps) {
    auto norm_v = torch::norm(v, 2, -1, true);
    return v / torch::clamp(norm_v, eps);
}

torch::Tensor TensorOperations::safe_acos(const torch::Tensor& x, float eps) {
    auto clamped_x = torch::clamp(x, -1.0f + eps, 1.0f - eps);
    return torch::acos(clamped_x);
}

// DeviceManager implementation
DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

void DeviceManager::set_device(torch::Device device) {
    device_ = device;
    if (device_.is_cuda()) {
        torch::cuda::set_device(device_);
    }
}

int DeviceManager::device_id() const {
    if (device_.is_cuda()) {
        return device_.index();
    }
    return -1;
}

torch::Tensor DeviceManager::ensure_device(const torch::Tensor& tensor) const {
    if (tensor.device() != device_) {
        return tensor.to(device_);
    }
    return tensor;
}

void DeviceManager::synchronize_if_cuda() const {
    if (device_.is_cuda()) {
        torch::cuda::synchronize(device_);
    }
}

size_t DeviceManager::get_available_memory() const {
    if (device_.is_cuda()) {
        auto props = torch::cuda::getDeviceProperties(device_);
        auto free_memory = torch::cuda::mem_get_info(device_);
        return free_memory.first;
    }
    // For CPU, return a large number or use system-specific calls
    return SIZE_MAX;
}

size_t DeviceManager::get_total_memory() const {
    if (device_.is_cuda()) {
        auto props = torch::cuda::getDeviceProperties(device_);
        return props.totalGlobalMem;
    }
    // For CPU, return a large number or use system-specific calls
    return SIZE_MAX;
}

// MemoryPool implementation
MemoryPool::MemoryPool(size_t max_pool_size) : max_pool_size_(max_pool_size) {}

torch::Tensor MemoryPool::get_tensor(const std::vector<int64_t>& shape,
                                    torch::ScalarType dtype,
                                    torch::Device device) {
    std::string key = tensor_key(shape, dtype, device);
    
    auto it = tensor_map_.find(key);
    if (it != tensor_map_.end() && it->second < tensor_pool_.size()) {
        auto& info = tensor_pool_[it->second];
        if (!info.in_use && info.shape == shape && info.dtype == dtype && info.device == device) {
            info.in_use = true;
            return info.tensor.view(shape);
        }
    }
    
    // Create new tensor
    auto tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device));
    
    if (tensor_pool_.size() < max_pool_size_) {
        TensorInfo info;
        info.tensor = tensor.flatten();
        info.shape = shape;
        info.dtype = dtype;
        info.device = device;
        info.in_use = true;
        
        tensor_pool_.push_back(info);
        tensor_map_[key] = tensor_pool_.size() - 1;
        
        return tensor;
    }
    
    return tensor;
}

void MemoryPool::return_tensor(const torch::Tensor& tensor) {
    auto flat_tensor = tensor.flatten();
    for (auto& info : tensor_pool_) {
        if (info.tensor.data_ptr() == flat_tensor.data_ptr()) {
            info.in_use = false;
            break;
        }
    }
}

void MemoryPool::clear_pool() {
    tensor_pool_.clear();
    tensor_map_.clear();
}

size_t MemoryPool::active_tensors() const {
    size_t count = 0;
    for (const auto& info : tensor_pool_) {
        if (info.in_use) count++;
    }
    return count;
}

std::string MemoryPool::tensor_key(const std::vector<int64_t>& shape,
                                  torch::ScalarType dtype,
                                  torch::Device device) const {
    std::string key = std::to_string(device.index()) + "_";
    key += std::to_string(static_cast<int>(dtype)) + "_";
    for (auto dim : shape) {
        key += std::to_string(dim) + "x";
    }
    return key;
}

} // namespace dna2