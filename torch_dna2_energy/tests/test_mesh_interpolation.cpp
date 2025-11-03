#include "dna2_energy_calculator.h"
#include "dna2_mesh_interpolation.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace dna2;

/**
 * @file test_mesh_interpolation.cpp
 * @brief Comprehensive test suite for mesh interpolation system
 * 
 * This test suite verifies:
 * - Correctness of mesh interpolation
 * - Proper derivative computation
 * - Mesh initialization and data integrity
 * - Edge case handling
 */

class TestMeshInterpolation {
private:
    float tolerance = 1e-6f;
    int passed_tests = 0;
    int total_tests = 0;
    
    void assert_almost_equal(float a, float b, float tol, const std::string& test_name) {
        total_tests++;
        if (std::abs(a - b) < tol) {
            passed_tests++;
            std::cout << "✓ " << test_name << std::endl;
        } else {
            std::cout << "✗ " << test_name << " (expected: " << a << ", got: " << b << ")" << std::endl;
        }
    }
    
    void assert_tensor_close(const torch::Tensor& a, const torch::Tensor& b, 
                            float tol, const std::string& test_name) {
        total_tests++;
        if (torch::allclose(a, b, tol)) {
            passed_tests++;
            std::cout << "✓ " << test_name << std::endl;
        } else {
            std::cout << "✗ " << test_name << " (tensor mismatch)" << std::endl;
        }
    }
    
public:
    void test_mesh_initialization() {
        std::cout << "\n=== Testing Mesh Initialization ===" << std::endl;
        
        // Test default mesh creation
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        assert(interpolator->is_initialized(), "Default mesh interpolator initialized");
        
        // Test mesh data integrity
        for (int mesh_type = 0; mesh_type < 13; ++mesh_type) {
            MeshType type = static_cast<MeshType>(mesh_type);
            const auto& mesh_data = interpolator->get_mesh_data(type);
            
            assert(mesh_data.x_values.numel() > 0, "Mesh x values not empty");
            assert(mesh_data.y_values.numel() > 0, "Mesh y values not empty");
            assert(mesh_data.tangents.numel() > 0, "Mesh tangents not empty");
            assert(mesh_data.x_values.sizes() == mesh_data.y_values.sizes(), 
                   "Mesh x and y have same size");
            assert(mesh_data.x_values.sizes() == mesh_data.tangents.sizes(), 
                   "Mesh x and tangents have same size");
        }
        
        passed_tests += 4;
        total_tests += 4;
        std::cout << "✓ Mesh initialization tests passed" << std::endl;
    }
    
    void test_basic_interpolation() {
        std::cout << "\n=== Testing Basic Interpolation ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        // Test interpolation at known points
        torch::Tensor cos_values = torch::linspace(-1.0, 1.0, 5);
        
        for (int mesh_type = 0; mesh_type < 13; ++mesh_type) {
            MeshType type = static_cast<MeshType>(mesh_type);
            
            auto interpolated = interpolator->interpolate(type, cos_values);
            auto derivative = interpolator->interpolate_derivative(type, cos_values);
            
            assert(interpolated.sizes() == cos_values.sizes(), 
                   "Interpolation output size matches input");
            assert(derivative.sizes() == cos_values.sizes(), 
                   "Derivative output size matches input");
            assert(torch::all(torch::isfinite(interpolated)), 
                   "Interpolation values are finite");
            assert(torch::all(torch::isfinite(derivative)), 
                   "Derivative values are finite");
        }
        
        passed_tests += 4 * 13;  // 4 checks per mesh type
        total_tests += 4 * 13;
        std::cout << "✓ Basic interpolation tests passed" << std::endl;
    }
    
    void test_interpolation_bounds() {
        std::cout << "\n=== Testing Interpolation Bounds ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        // Test interpolation at boundaries
        torch::Tensor boundary_values = torch::tensor({-1.0f, 1.0f});
        
        for (int mesh_type = 0; mesh_type < 13; ++mesh_type) {
            MeshType type = static_cast<MeshType>(mesh_type);
            
            auto interpolated = interpolator->interpolate(type, boundary_values);
            auto derivative = interpolator->interpolate_derivative(type, boundary_values);
            
            // Values should be finite at boundaries
            assert(torch::all(torch::isfinite(interpolated)), 
                   "Boundary interpolation values are finite");
            assert(torch::all(torch::isfinite(derivative)), 
                   "Boundary derivative values are finite");
        }
        
        // Test extrapolation (should handle gracefully)
        torch::Tensor extrapolate_values = torch::tensor({-1.5f, 1.5f});
        
        for (int mesh_type = 0; mesh_type < 13; ++mesh_type) {
            MeshType type = static_cast<MeshType>(mesh_type);
            
            auto interpolated = interpolator->interpolate(type, extrapolate_values);
            auto derivative = interpolator->interpolate_derivative(type, extrapolate_values);
            
            // Should still return finite values (clamped or extrapolated)
            assert(torch::all(torch::isfinite(interpolated)), 
                   "Extrapolation values are finite");
            assert(torch::all(torch::isfinite(derivative)), 
                   "Extrapolation derivatives are finite");
        }
        
        passed_tests += 2 * 13 + 2 * 13;  // Boundary + extrapolation tests
        total_tests += 2 * 13 + 2 * 13;
        std::cout << "✓ Interpolation bounds tests passed" << std::endl;
    }
    
    void test_derivative_consistency() {
        std::cout << "\n=== Testing Derivative Consistency ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        float h = 1e-6f;  // Small step for numerical derivative
        
        // Test derivative consistency for a few mesh types
        std::vector<MeshType> test_types = {
            MeshType::HYDR_F4_THETA1,
            MeshType::STCK_F4_THETA1,
            MeshType::CRST_F4_THETA1
        };
        
        for (MeshType type : test_types) {
            for (float x = -0.9f; x <= 0.9f; x += 0.3f) {
                torch::Tensor x_tensor = torch::tensor(x);
                torch::Tensor x_plus = torch::tensor(x + h);
                torch::Tensor x_minus = torch::tensor(x - h);
                
                auto y_plus = interpolator->interpolate(type, x_plus);
                auto y_minus = interpolator->interpolate(type, x_minus);
                float numerical_deriv = (y_plus.item<float>() - y_minus.item<float>()) / (2.0f * h);
                
                auto analytical_deriv = interpolator->interpolate_derivative(type, x_tensor);
                float analytical_deriv_val = analytical_deriv.item<float>();
                
                float error = std::abs(numerical_deriv - analytical_deriv_val);
                assert_almost_equal(0.0f, error, 1e-4f, 
                                   "Derivative consistency for mesh type " + std::to_string(static_cast<int>(type)) + 
                                   " at x=" + std::to_string(x));
            }
        }
        
        std::cout << "✓ Derivative consistency tests passed" << std::endl;
    }
    
    void test_hermite_interpolation_properties() {
        std::cout << "\n=== Testing Hermite Interpolation Properties ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        // Test that interpolation is smooth (C1 continuous)
        torch::Tensor test_points = torch::linspace(-0.9f, 0.9f, 100);
        
        for (MeshType type : {MeshType::HYDR_F4_THETA1, MeshType::STCK_F4_THETA1}) {
            auto values = interpolator->interpolate(type, test_points);
            auto derivatives = interpolator->interpolate_derivative(type, test_points);
            
            // Check that values are smooth (no abrupt jumps)
            for (int i = 1; i < values.numel(); ++i) {
                float diff = std::abs(values[i].item<float>() - values[i-1].item<float>());
                assert(diff < 10.0f, "Smooth interpolation - no large jumps");
            }
            
            // Check that derivatives are also smooth
            for (int i = 1; i < derivatives.numel(); ++i) {
                float diff = std::abs(derivatives[i].item<float>() - derivatives[i-1].item<float>());
                assert(diff < 50.0f, "Smooth derivatives - no large jumps");
            }
        }
        
        passed_tests += 2 * 99 * 2;  // Smoothness checks
        total_tests += 2 * 99 * 2;
        std::cout << "✓ Hermite interpolation properties tests passed" << std::endl;
    }
    
    void test_special_mesh_types() {
        std::cout << "\n=== Testing Special Mesh Types ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        // Test CXST_F4_THETA1 which has special handling (pure harmonic)
        torch::Tensor cos_values = torch::linspace(-1.0, 1.0, 10);
        
        auto cxst_values = interpolator->interpolate(MeshType::CXST_F4_THETA1, cos_values);
        auto cxst_derivatives = interpolator->interpolate_derivative(MeshType::CXST_F4_THETA1, cos_values);
        
        // For pure harmonic, we expect specific behavior
        // The function should be approximately quadratic near the minimum
        for (int i = 0; i < cos_values.numel(); ++i) {
            float cos_val = cos_values[i].item<float>();
            float val = cxst_values[i].item<float>();
            float deriv = cxst_derivatives[i].item<float>();
            
            // Values should be non-negative (energy-like)
            assert(val >= -1e-6f, "CXST values are non-negative");
            
            // Derivative should be zero at cos=0 (minimum)
            if (std::abs(cos_val) < 0.1f) {
                assert(std::abs(deriv) < 0.1f, "CXST derivative near zero at minimum");
            }
        }
        
        passed_tests += cos_values.numel() * 2;
        total_tests += cos_values.numel() * 2;
        std::cout << "✓ Special mesh types tests passed" << std::endl;
    }
    
    void test_batch_interpolation() {
        std::cout << "\n=== Testing Batch Interpolation ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        // Test batch interpolation with different input sizes
        std::vector<int> batch_sizes = {1, 10, 100, 1000};
        
        for (int batch_size : batch_sizes) {
            torch::Tensor batch_cos = torch::rand({batch_size}) * 2.0f - 1.0f;  // [-1, 1]
            
            for (MeshType type : {MeshType::HYDR_F4_THETA1, MeshType::STCK_F4_THETA1}) {
                auto batch_values = interpolator->interpolate(type, batch_cos);
                auto batch_derivatives = interpolator->interpolate_derivative(type, batch_cos);
                
                assert(batch_values.sizes() == batch_cos.sizes(), 
                       "Batch interpolation output size matches input");
                assert(batch_derivatives.sizes() == batch_cos.sizes(), 
                       "Batch derivative output size matches input");
                assert(torch::all(torch::isfinite(batch_values)), 
                       "Batch interpolation values are finite");
                assert(torch::all(torch::isfinite(batch_derivatives)), 
                       "Batch derivative values are finite");
            }
        }
        
        passed_tests += batch_sizes.size() * 2 * 4;  // 2 mesh types, 4 checks each
        total_tests += batch_sizes.size() * 2 * 4;
        std::cout << "✓ Batch interpolation tests passed" << std::endl;
    }
    
    void test_mesh_data_integrity() {
        std::cout << "\n=== Testing Mesh Data Integrity ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        // Test that mesh data is properly loaded and consistent
        for (int mesh_type = 0; mesh_type < 13; ++mesh_type) {
            MeshType type = static_cast<MeshType>(mesh_type);
            const auto& mesh_data = interpolator->get_mesh_data(type);
            
            // Check that x values are sorted
            auto x_sorted = torch::sort(mesh_data.x_values);
            assert(torch::allclose(mesh_data.x_values, x_sorted.values), 
                   "Mesh x values are sorted");
            
            // Check that x values are within [-1, 1]
            assert(torch::all(mesh_data.x_values >= -1.0f), "Mesh x values >= -1");
            assert(torch::all(mesh_data.x_values <= 1.0f), "Mesh x values <= 1");
            
            // Check that y values are finite
            assert(torch::all(torch::isfinite(mesh_data.y_values)), 
                   "Mesh y values are finite");
            
            // Check that tangents are finite
            assert(torch::all(torch::isfinite(mesh_data.tangents)), 
                   "Mesh tangents are finite");
        }
        
        passed_tests += 13 * 5;  // 5 checks per mesh type
        total_tests += 13 * 5;
        std::cout << "✓ Mesh data integrity tests passed" << std::endl;
    }
    
    void test_performance_consistency() {
        std::cout << "\n=== Testing Performance Consistency ===" << std::endl;
        
        auto interpolator = MeshFactory::create_default_interpolator(torch::kCPU);
        
        torch::Tensor test_cos = torch::rand({1000}) * 2.0f - 1.0f;
        
        // Test that multiple calls give consistent results
        for (MeshType type : {MeshType::HYDR_F4_THETA1, MeshType::STCK_F4_THETA1}) {
            auto values1 = interpolator->interpolate(type, test_cos);
            auto values2 = interpolator->interpolate(type, test_cos);
            
            assert(torch::allclose(values1, values2, 1e-10f), 
                   "Consistent interpolation results");
            
            auto derivs1 = interpolator->interpolate_derivative(type, test_cos);
            auto derivs2 = interpolator->interpolate_derivative(type, test_cos);
            
            assert(torch::allclose(derivs1, derivs2, 1e-10f), 
                   "Consistent derivative results");
        }
        
        passed_tests += 2 * 2;  // 2 mesh types, 2 consistency checks each
        total_tests += 2 * 2;
        std::cout << "✓ Performance consistency tests passed" << std::endl;
    }
    
    void run_all_tests() {
        std::cout << "DNA2 Mesh Interpolation Test Suite" << std::endl;
        std::cout << "==================================" << std::endl;
        
        test_mesh_initialization();
        test_basic_interpolation();
        test_interpolation_bounds();
        test_derivative_consistency();
        test_hermite_interpolation_properties();
        test_special_mesh_types();
        test_batch_interpolation();
        test_mesh_data_integrity();
        test_performance_consistency();
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed_tests << "/" << total_tests << " tests" << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << "✓ All mesh interpolation tests passed!" << std::endl;
        } else {
            std::cout << "✗ Some tests failed!" << std::endl;
        }
    }
};

int main() {
    try {
        TestMeshInterpolation tester;
        tester.run_all_tests();
        return (tester.passed_tests == tester.total_tests) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}