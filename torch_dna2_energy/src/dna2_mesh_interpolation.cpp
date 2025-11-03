#include "dna2_mesh_interpolation.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace dna2 {

// MeshData implementation
MeshData::MeshData(int type, int n_pts, torch::Device device) 
    : mesh_type(type), n_points(n_pts) {
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    x_values = torch::zeros({n_pts}, options);
    y_values = torch::zeros({n_pts}, options);
    y_derivatives = torch::zeros({n_pts}, options);
    tangents = torch::zeros({n_pts}, options);
}

void MeshData::to(torch::Device new_device) {
    x_values = x_values.to(new_device);
    y_values = y_values.to(new_device);
    y_derivatives = y_derivatives.to(new_device);
    tangents = tangents.to(new_device);
}

// MeshInterpolator implementation
MeshInterpolator::MeshInterpolator(torch::Device device) 
    : initialized_(false), device_(device) {
}

void MeshInterpolator::initialize_meshes(const DNA2Parameters& params) {
    if (initialized_) {
        return;
    }
    
    initialize_hydr_meshes(params);
    initialize_stck_meshes(params);
    initialize_crst_meshes(params);
    initialize_cxst_meshes(params);
    
    initialized_ = true;
}

void MeshInterpolator::initialize_hydr_meshes(const DNA2Parameters& params) {
    // HYDR_F4_THETA1
    initialize_single_mesh(MeshType::HYDR_F4_THETA1, params, HYDR_T1_MESH_POINTS,
                          HYDR_THETA1_T0, HYDR_THETA1_TC, HYDR_THETA1_TS);
    
    // HYDR_F4_THETA2
    initialize_single_mesh(MeshType::HYDR_F4_THETA2, params, HYDR_T2_MESH_POINTS,
                          HYDR_THETA2_T0, HYDR_THETA2_TC, HYDR_THETA2_TS);
    
    // HYDR_F4_THETA3
    initialize_single_mesh(MeshType::HYDR_F4_THETA3, params, HYDR_T3_MESH_POINTS,
                          HYDR_THETA3_T0, HYDR_THETA3_TC, HYDR_THETA3_TS);
    
    // HYDR_F4_THETA4
    initialize_single_mesh(MeshType::HYDR_F4_THETA4, params, HYDR_T4_MESH_POINTS,
                          HYDR_THETA4_T0, HYDR_THETA4_TC, HYDR_THETA4_TS);
    
    // HYDR_F4_THETA7
    initialize_single_mesh(MeshType::HYDR_F4_THETA7, params, HYDR_T7_MESH_POINTS,
                          HYDR_THETA7_T0, HYDR_THETA7_TC, HYDR_THETA7_TS);
    
    // HYDR_F4_THETA8
    initialize_single_mesh(MeshType::HYDR_F4_THETA8, params, HYDR_T8_MESH_POINTS,
                          HYDR_THETA8_T0, HYDR_THETA8_TC, HYDR_THETA8_TS);
}

void MeshInterpolator::initialize_stck_meshes(const DNA2Parameters& params) {
    // STCK_F4_THETA1
    initialize_single_mesh(MeshType::STCK_F4_THETA1, params, STCK_T1_MESH_POINTS,
                          STCK_THETA1_T0, STCK_THETA1_TC, STCK_THETA1_TS);
    
    // STCK_F4_THETA2
    initialize_single_mesh(MeshType::STCK_F4_THETA2, params, STCK_T2_MESH_POINTS,
                          STCK_THETA2_T0, STCK_THETA2_TC, STCK_THETA2_TS);
    
    // STCK_F4_THETA3
    initialize_single_mesh(MeshType::STCK_F4_THETA3, params, STCK_T3_MESH_POINTS,
                          STCK_THETA3_T0, STCK_THETA3_TC, STCK_THETA3_TS);
    
    // STCK_F4_THETA4
    initialize_single_mesh(MeshType::STCK_F4_THETA4, params, STCK_T4_MESH_POINTS,
                          STCK_THETA4_T0, STCK_THETA4_TC, STCK_THETA4_TS);
}

void MeshInterpolator::initialize_crst_meshes(const DNA2Parameters& params) {
    // CRST_F4_THETA1
    initialize_single_mesh(MeshType::CRST_F4_THETA1, params, CRST_T1_MESH_POINTS,
                          CRST_THETA1_T0, CRST_THETA1_TC, CRST_THETA1_TS);
    
    // CRST_F4_THETA2
    initialize_single_mesh(MeshType::CRST_F4_THETA2, params, CRST_T2_MESH_POINTS,
                          CRST_THETA2_T0, CRST_THETA2_TC, CRST_THETA2_TS);
}

void MeshInterpolator::initialize_cxst_meshes(const DNA2Parameters& params) {
    // CXST_F4_THETA1 - special case with pure harmonic addition
    initialize_cxst_theta1_mesh(params);
}

void MeshInterpolator::initialize_single_mesh(MeshType mesh_type, const DNA2Parameters& params,
                                            int n_points, float t0, float tc, float ts) {
    // Generate cosine values
    float upplimit = std::cos(std::max(0.0f, t0 - tc));
    float lowlimit = std::cos(std::min(M_PI, t0 + tc));
    
    auto x_values = torch::linspace(lowlimit, upplimit, n_points,
                                   torch::TensorOptions().device(device_));
    
    // Generate function values
    auto theta_values = torch::acos(x_values);
    auto y_values = compute_f4_function(theta_values, mesh_type, params);
    auto y_derivatives = compute_f4_derivative(theta_values, mesh_type, params);
    
    // Compute Hermite tangents
    auto tangents = compute_hermite_tangents(x_values, y_values, y_derivatives);
    
    // Store mesh data
    MeshData mesh(static_cast<int>(mesh_type), n_points, device_);
    mesh.x_values = x_values;
    mesh.y_values = y_values;
    mesh.y_derivatives = y_derivatives;
    mesh.tangents = tangents;
    
    meshes_[static_cast<int>(mesh_type)] = mesh;
}

void MeshInterpolator::initialize_cxst_theta1_mesh(const DNA2Parameters& params) {
    // CXST_F4_THETA1 uses pure harmonic addition
    int n_points = CXST_T1_MESH_POINTS;
    float t0 = CXST_THETA1_T0;
    float tc = CXST_THETA1_TC;
    float ts = CXST_THETA1_TS;
    
    // Generate cosine values
    float upplimit = std::cos(std::max(0.0f, t0 - tc));
    float lowlimit = std::cos(std::min(M_PI, t0 + tc));
    
    auto x_values = torch::linspace(lowlimit, upplimit, n_points,
                                   torch::TensorOptions().device(device_));
    
    // For CXST_F4_THETA1, use pure harmonic function
    auto theta_values = torch::acos(x_values);
    auto dt = theta_values - t0;
    auto abs_dt = torch::abs(dt);
    
    // Pure harmonic: 1 - a * (theta - t0)^2
    float a = 1.0f / (ts * ts);
    auto y_values = 1.0f - a * torch::square(dt);
    auto y_derivatives = -2.0f * a * dt * torch::sign(dt);
    
    // Compute Hermite tangents
    auto tangents = compute_hermite_tangents(x_values, y_values, y_derivatives);
    
    // Store mesh data
    MeshData mesh(static_cast<int>(MeshType::CXST_F4_THETA1), n_points, device_);
    mesh.x_values = x_values;
    mesh.y_values = y_values;
    mesh.y_derivatives = y_derivatives;
    mesh.tangents = tangents;
    
    meshes_[static_cast<int>(MeshType::CXST_F4_THETA1)] = mesh;
}

torch::Tensor MeshInterpolator::interpolate(MeshType mesh_type, const torch::Tensor& cos_values) {
    if (!initialized_) {
        throw DNA2EnergyException("Mesh interpolator not initialized");
    }
    
    const auto& mesh = meshes_[static_cast<int>(mesh_type)];
    
    // Find interpolation intervals
    auto indices = torch::searchsorted(mesh.x_values, cos_values);
    indices = torch::clamp(indices, 1, static_cast<int>(mesh.x_values.size(0)) - 1);
    
    // Get interval endpoints
    auto i0 = indices - 1;
    auto i1 = indices;
    
    auto x0 = mesh.x_values.index({i0});
    auto x1 = mesh.x_values.index({i1});
    auto y0 = mesh.y_values.index({i0});
    auto y1 = mesh.y_values.index({i1});
    auto m0 = mesh.tangents.index({i0});
    auto m1 = mesh.tangents.index({i1});
    
    // Compute interpolation parameter
    auto t = (cos_values - x0) / (x1 - x0);
    auto t2 = t * t;
    auto t3 = t2 * t;
    
    // Hermite basis functions
    auto h00 = 2*t3 - 3*t2 + 1;
    auto h10 = t3 - 2*t2 + t;
    auto h01 = -2*t3 + 3*t2;
    auto h11 = t3 - t2;
    
    // Interpolated value
    return h00*y0 + h10*(x1-x0)*m0 + h01*y1 + h11*(x1-x0)*m1;
}

torch::Tensor MeshInterpolator::interpolate_derivative(MeshType mesh_type, const torch::Tensor& cos_values) {
    if (!initialized_) {
        throw DNA2EnergyException("Mesh interpolator not initialized");
    }
    
    const auto& mesh = meshes_[static_cast<int>(mesh_type)];
    
    // Find interpolation intervals (same as interpolate)
    auto indices = torch::searchsorted(mesh.x_values, cos_values);
    indices = torch::clamp(indices, 1, static_cast<int>(mesh.x_values.size(0)) - 1);
    
    auto i0 = indices - 1;
    auto i1 = indices;
    
    auto x0 = mesh.x_values.index({i0});
    auto x1 = mesh.x_values.index({i1});
    auto y0 = mesh.y_values.index({i0});
    auto y1 = mesh.y_values.index({i1});
    auto m0 = mesh.tangents.index({i0});
    auto m1 = mesh.tangents.index({i1});
    
    // Compute interpolation parameter
    auto t = (cos_values - x0) / (x1 - x0);
    auto t2 = t * t;
    
    // Derivative of Hermite basis functions
    auto dh00 = 6*t2 - 6*t;
    auto dh10 = 3*t2 - 4*t + 1;
    auto dh01 = -6*t2 + 6*t;
    auto dh11 = 3*t2 - 2*t;
    
    // Interpolated derivative
    return (dh00*y0 + dh10*(x1-x0)*m0 + dh01*y1 + dh11*(x1-x0)*m1) / (x1 - x0);
}

const MeshData& MeshInterpolator::get_mesh_data(MeshType mesh_type) const {
    if (!initialized_) {
        throw DNA2EnergyException("Mesh interpolator not initialized");
    }
    
    auto it = meshes_.find(static_cast<int>(mesh_type));
    if (it == meshes_.end()) {
        throw DNA2EnergyException("Mesh type not found");
    }
    
    return it->second;
}

void MeshInterpolator::to(torch::Device new_device) {
    device_ = new_device;
    for (auto& pair : meshes_) {
        pair.second.to(new_device);
    }
}

torch::Tensor MeshInterpolator::compute_hermite_tangents(const torch::Tensor& x,
                                                        const torch::Tensor& y,
                                                        const torch::Tensor& dy) {
    auto n = x.size(0);
    auto tangents = torch::zeros_like(y);
    
    // Internal points - use Catmull-Rom tangents
    if (n > 2) {
        auto dx = x.slice(0, 2) - x.slice(0, 0, n-2);
        auto dy_dx = dy.slice(0, 1, n-1);
        
        tangents.slice(0, 1, n-1) = 0.5 * (
            (y.slice(0, 2) - y.slice(0, 0, n-2)) / dx.unsqueeze(-1) +
            dy_dx
        );
    }
    
    // Boundary points - use one-sided differences
    if (n > 1) {
        tangents[0] = (y[1] - y[0]) / (x[1] - x[0]);
        tangents[n-1] = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]);
    }
    
    return tangents;
}

torch::Tensor MeshInterpolator::compute_f4_function(const torch::Tensor& theta, MeshType mesh_type,
                                                   const DNA2Parameters& params) {
    // Get parameters based on mesh type
    float t0, tc, ts, a, b;
    
    switch (mesh_type) {
        case MeshType::HYDR_F4_THETA1:
            t0 = HYDR_THETA1_T0; tc = HYDR_THETA1_TC; ts = HYDR_THETA1_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA2:
            t0 = HYDR_THETA2_T0; tc = HYDR_THETA2_TC; ts = HYDR_THETA2_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA3:
            t0 = HYDR_THETA3_T0; tc = HYDR_THETA3_TC; ts = HYDR_THETA3_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA4:
            t0 = HYDR_THETA4_T0; tc = HYDR_THETA4_TC; ts = HYDR_THETA4_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA7:
            t0 = HYDR_THETA7_T0; tc = HYDR_THETA7_TC; ts = HYDR_THETA7_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA8:
            t0 = HYDR_THETA8_T0; tc = HYDR_THETA8_TC; ts = HYDR_THETA8_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA1:
            t0 = STCK_THETA1_T0; tc = STCK_THETA1_TC; ts = STCK_THETA1_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA2:
            t0 = STCK_THETA2_T0; tc = STCK_THETA2_TC; ts = STCK_THETA2_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA3:
            t0 = STCK_THETA3_T0; tc = STCK_THETA3_TC; ts = STCK_THETA3_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA4:
            t0 = STCK_THETA4_T0; tc = STCK_THETA4_TC; ts = STCK_THETA4_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::CRST_F4_THETA1:
            t0 = CRST_THETA1_T0; tc = CRST_THETA1_TC; ts = CRST_THETA1_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::CRST_F4_THETA2:
            t0 = CRST_THETA2_T0; tc = CRST_THETA2_TC; ts = CRST_THETA2_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        default:
            throw DNA2EnergyException("Unknown mesh type for f4 function");
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

torch::Tensor MeshInterpolator::compute_f4_derivative(const torch::Tensor& theta, MeshType mesh_type,
                                                     const DNA2Parameters& params) {
    // Get parameters based on mesh type
    float t0, tc, ts, a, b;
    
    switch (mesh_type) {
        case MeshType::HYDR_F4_THETA1:
            t0 = HYDR_THETA1_T0; tc = HYDR_THETA1_TC; ts = HYDR_THETA1_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA2:
            t0 = HYDR_THETA2_T0; tc = HYDR_THETA2_TC; ts = HYDR_THETA2_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA3:
            t0 = HYDR_THETA3_T0; tc = HYDR_THETA3_TC; ts = HYDR_THETA3_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA4:
            t0 = HYDR_THETA4_T0; tc = HYDR_THETA4_TC; ts = HYDR_THETA4_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA7:
            t0 = HYDR_THETA7_T0; tc = HYDR_THETA7_TC; ts = HYDR_THETA7_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::HYDR_F4_THETA8:
            t0 = HYDR_THETA8_T0; tc = HYDR_THETA8_TC; ts = HYDR_THETA8_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA1:
            t0 = STCK_THETA1_T0; tc = STCK_THETA1_TC; ts = STCK_THETA1_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA2:
            t0 = STCK_THETA2_T0; tc = STCK_THETA2_TC; ts = STCK_THETA2_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA3:
            t0 = STCK_THETA3_T0; tc = STCK_THETA3_TC; ts = STCK_THETA3_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::STCK_F4_THETA4:
            t0 = STCK_THETA4_T0; tc = STCK_THETA4_TC; ts = STCK_THETA4_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::CRST_F4_THETA1:
            t0 = CRST_THETA1_T0; tc = CRST_THETA1_TC; ts = CRST_THETA1_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        case MeshType::CRST_F4_THETA2:
            t0 = CRST_THETA2_T0; tc = CRST_THETA2_TC; ts = CRST_THETA2_TS;
            a = 0.5f / (ts * ts); b = 0.5f / ((tc - ts) * (tc - ts));
            break;
        default:
            throw DNA2EnergyException("Unknown mesh type for f4 derivative");
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

// MeshFactory implementation
std::unique_ptr<MeshInterpolator> MeshFactory::create_default_interpolator(torch::Device device) {
    DNA2Parameters params;
    params.device = device;
    params.initialize_tensors();
    
    auto interpolator = std::make_unique<MeshInterpolator>(device);
    interpolator->initialize_meshes(params);
    
    return interpolator;
}

std::unique_ptr<MeshInterpolator> MeshFactory::create_interpolator(
    const DNA2Parameters& params,
    torch::Device device) {
    
    auto interpolator = std::make_unique<MeshInterpolator>(device);
    interpolator->initialize_meshes(params);
    
    return interpolator;
}

} // namespace dna2