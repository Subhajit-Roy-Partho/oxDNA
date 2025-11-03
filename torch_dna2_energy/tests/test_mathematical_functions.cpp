#include "dna2_energy_calculator.h"
#include "dna2_mathematical_functions.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace dna2;

/**
 * @file test_mathematical_functions.cpp
 * @brief Comprehensive test suite for mathematical functions
 * 
 * This test suite verifies:
 * - Mathematical accuracy of all functions
 * - Correctness of derivatives
 * - Edge case handling
 * - Numerical stability
 */

class TestMathematicalFunctions {
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
    void test_f1_function() {
        std::cout << "\n=== Testing f1 Function ===" << std::endl;
        
        DNA2Parameters params;
        params.initialize_tensors();
        
        // Test basic values
        float r = 0.5f;
        float r0 = 0.4f;
        float a = 8.0f;
        
        torch::Tensor r_tensor = torch::tensor(r);
        torch::Tensor r0_tensor = torch::tensor(r0);
        torch::Tensor a_tensor = torch::tensor(a);
        
        // Test f1 function
        torch::Tensor f1_val = MathematicalFunctions::f1(r_tensor, r0_tensor, a_tensor);
        float expected_f1 = (r < r0) ? std::pow(1.0f - std::pow(r/r0, 2), 2) : 0.0f;
        
        assert_almost_equal(expected_f1, f1_val.item<float>(), tolerance, 
                           "f1 basic value");
        
        // Test derivative
        torch::Tensor f1_deriv = MathematicalFunctions::f1_derivative(r_tensor, r0_tensor, a_tensor);
        float expected_deriv = (r < r0) ? -4.0f * r / (r0 * r0) * (1.0f - std::pow(r/r0, 2)) : 0.0f;
        
        assert_almost_equal(expected_deriv, f1_deriv.item<float>(), tolerance, 
                           "f1 derivative");
        
        // Test edge cases
        torch::Tensor f1_zero = MathematicalFunctions::f1(torch::tensor(0.0f), r0_tensor, a_tensor);
        assert_almost_equal(1.0f, f1_zero.item<float>(), tolerance, "f1 at r=0");
        
        torch::Tensor f1_at_r0 = MathematicalFunctions::f1(r0_tensor, r0_tensor, a_tensor);
        assert_almost_equal(0.0f, f1_at_r0.item<float>(), tolerance, "f1 at r=r0");
        
        torch::Tensor f1_beyond = MathematicalFunctions::f1(torch::tensor(r0 * 2), r0_tensor, a_tensor);
        assert_almost_equal(0.0f, f1_beyond.item<float>(), tolerance, "f1 beyond cutoff");
    }
    
    void test_f2_function() {
        std::cout << "\n=== Testing f2 Function ===" << std::endl;
        
        DNA2Parameters params;
        params.initialize_tensors();
        
        float r = 0.5f;
        float r0 = 0.4f;
        float a = 6.0f;
        
        torch::Tensor r_tensor = torch::tensor(r);
        torch::Tensor r0_tensor = torch::tensor(r0);
        torch::Tensor a_tensor = torch::tensor(a);
        
        // Test f2 function
        torch::Tensor f2_val = MathematicalFunctions::f2(r_tensor, r0_tensor, a_tensor);
        float expected_f2 = (r < r0) ? std::pow(1.0f - std::pow(r/r0, 2), 2) : 0.0f;
        
        assert_almost_equal(expected_f2, f2_val.item<float>(), tolerance, 
                           "f2 basic value");
        
        // Test derivative
        torch::Tensor f2_deriv = MathematicalFunctions::f2_derivative(r_tensor, r0_tensor, a_tensor);
        float expected_deriv = (r < r0) ? -4.0f * r / (r0 * r0) * (1.0f - std::pow(r/r0, 2)) : 0.0f;
        
        assert_almost_equal(expected_deriv, f2_deriv.item<float>(), tolerance, 
                           "f2 derivative");
    }
    
    void test_f4_function() {
        std::cout << "\n=== Testing f4 Function ===" << std::endl;
        
        DNA2Parameters params;
        params.initialize_tensors();
        
        float theta = M_PI / 3.0f;  // 60 degrees
        float theta0 = M_PI / 2.0f;  // 90 degrees
        float a = 6.0f;
        
        torch::Tensor theta_tensor = torch::tensor(theta);
        torch::Tensor theta0_tensor = torch::tensor(theta0);
        torch::Tensor a_tensor = torch::tensor(a);
        
        // Test f4 function
        torch::Tensor f4_val = MathematicalFunctions::f4(theta_tensor, theta0_tensor, a_tensor);
        float cos_theta = std::cos(theta);
        float cos_theta0 = std::cos(theta0);
        float expected_f4 = std::pow((cos_theta - cos_theta0) / (1.0f - cos_theta0), 2);
        
        assert_almost_equal(expected_f4, f4_val.item<float>(), tolerance, 
                           "f4 basic value");
        
        // Test derivative
        torch::Tensor f4_deriv = MathematicalFunctions::f4_derivative(theta_tensor, theta0_tensor, a_tensor);
        float sin_theta = std::sin(theta);
        float expected_deriv = -2.0f * sin_theta * (cos_theta - cos_theta0) / 
                              std::pow(1.0f - cos_theta0, 2);
        
        assert_almost_equal(expected_deriv, f4_deriv.item<float>(), tolerance, 
                           "f4 derivative");
        
        // Test edge cases
        torch::Tensor f4_at_theta0 = MathematicalFunctions::f4(theta0_tensor, theta0_tensor, a_tensor);
        assert_almost_equal(0.0f, f4_at_theta0.item<float>(), tolerance, "f4 at theta=theta0");
        
        torch::Tensor f4_at_pi = MathematicalFunctions::f4(torch::tensor(M_PI), theta0_tensor, a_tensor);
        float expected_at_pi = std::pow((-1.0f - cos_theta0) / (1.0f - cos_theta0), 2);
        assert_almost_equal(expected_at_pi, f4_at_pi.item<float>(), tolerance, "f4 at theta=pi");
    }
    
    void test_f5_function() {
        std::cout << "\n=== Testing f5 Function ===" << std::endl;
        
        DNA2Parameters params;
        params.initialize_tensors();
        
        float phi = M_PI / 4.0f;  // 45 degrees
        float phi0 = M_PI / 2.0f;  // 90 degrees
        float a = 1.0f;
        
        torch::Tensor phi_tensor = torch::tensor(phi);
        torch::Tensor phi0_tensor = torch::tensor(phi0);
        torch::Tensor a_tensor = torch::tensor(a);
        
        // Test f5 function
        torch::Tensor f5_val = MathematicalFunctions::f5(phi_tensor, phi0_tensor, a_tensor);
        float expected_f5 = 1.0f - 0.5f * (1.0f + std::cos(phi - phi0));
        
        assert_almost_equal(expected_f5, f5_val.item<float>(), tolerance, 
                           "f5 basic value");
        
        // Test derivative
        torch::Tensor f5_deriv = MathematicalFunctions::f5_derivative(phi_tensor, phi0_tensor, a_tensor);
        float expected_deriv = 0.5f * std::sin(phi - phi0);
        
        assert_almost_equal(expected_deriv, f5_deriv.item<float>(), tolerance, 
                           "f5 derivative");
        
        // Test edge cases
        torch::Tensor f5_at_phi0 = MathematicalFunctions::f5(phi0_tensor, phi0_tensor, a_tensor);
        assert_almost_equal(0.5f, f5_at_phi0.item<float>(), tolerance, "f5 at phi=phi0");
    }
    
    void test_tensor_operations() {
        std::cout << "\n=== Testing Tensor Operations ===" << std::endl;
        
        // Test cross product
        torch::Tensor a = torch::tensor({1.0f, 0.0f, 0.0f});
        torch::Tensor b = torch::tensor({0.0f, 1.0f, 0.0f});
        torch::Tensor cross = TensorOperations::cross_product(a, b);
        torch::Tensor expected_cross = torch::tensor({0.0f, 0.0f, 1.0f});
        
        assert_tensor_close(cross, expected_cross, tolerance, "cross product");
        
        // Test dot product
        torch::Tensor dot = TensorOperations::dot_product(a, b);
        assert_almost_equal(0.0f, dot.item<float>(), tolerance, "dot product (orthogonal)");
        
        torch::Tensor c = torch::tensor({1.0f, 2.0f, 3.0f});
        torch::Tensor d = torch::tensor({4.0f, 5.0f, 6.0f});
        torch::Tensor dot2 = TensorOperations::dot_product(c, d);
        assert_almost_equal(32.0f, dot2.item<float>(), tolerance, "dot product (general)");
        
        // Test normalization
        torch::Tensor norm_a = TensorOperations::normalize(a);
        assert_tensor_close(norm_a, a, tolerance, "normalize unit vector");
        
        torch::Tensor e = torch::tensor({3.0f, 4.0f, 0.0f});
        torch::Tensor norm_e = TensorOperations::normalize(e);
        torch::Tensor expected_norm_e = torch::tensor({0.6f, 0.8f, 0.0f});
        assert_tensor_close(norm_e, expected_norm_e, tolerance, "normalize general vector");
        
        // Test matrix operations
        torch::Tensor mat = torch::tensor({{1.0f, 2.0f, 3.0f},
                                          {4.0f, 5.0f, 6.0f},
                                          {7.0f, 8.0f, 9.0f}});
        torch::Tensor mat_trans = TensorOperations::transpose(mat);
        torch::Tensor expected_trans = torch::tensor({{1.0f, 4.0f, 7.0f},
                                                     {2.0f, 5.0f, 8.0f},
                                                     {3.0f, 6.0f, 9.0f}});
        assert_tensor_close(mat_trans, expected_trans, tolerance, "matrix transpose");
    }
    
    void test_geometric_functions() {
        std::cout << "\n=== Testing Geometric Functions ===" << std::endl;
        
        // Test angle calculation
        torch::Tensor v1 = torch::tensor({1.0f, 0.0f, 0.0f});
        torch::Tensor v2 = torch::tensor({0.0f, 1.0f, 0.0f});
        torch::Tensor angle = TensorOperations::angle_between(v1, v2);
        assert_almost_equal(M_PI / 2.0f, angle.item<float>(), tolerance, "angle 90 degrees");
        
        torch::Tensor v3 = torch::tensor({1.0f, 0.0f, 0.0f});
        torch::Tensor angle2 = TensorOperations::angle_between(v1, v3);
        assert_almost_equal(0.0f, angle2.item<float>(), tolerance, "angle 0 degrees");
        
        // Test dihedral calculation
        torch::Tensor p1 = torch::tensor({0.0f, 0.0f, 0.0f});
        torch::Tensor p2 = torch::tensor({1.0f, 0.0f, 0.0f});
        torch::Tensor p3 = torch::tensor({1.0f, 1.0f, 0.0f});
        torch::Tensor p4 = torch::tensor({1.0f, 1.0f, 1.0f});
        torch::Tensor dihedral = TensorOperations::dihedral_angle(p1, p2, p3, p4);
        
        // This should be approximately 90 degrees for this configuration
        assert_almost_equal(M_PI / 2.0f, dihedral.item<float>(), 0.1f, "dihedral angle");
        
        // Test distance calculations
        torch::Tensor dist = TensorOperations::distance(p1, p4);
        assert_almost_equal(std::sqrt(3.0f), dist.item<float>(), tolerance, "distance calculation");
        
        // Test batch distance calculation
        torch::Tensor positions = torch::tensor({{0.0f, 0.0f, 0.0f},
                                                {1.0f, 0.0f, 0.0f},
                                                {0.0f, 1.0f, 0.0f}});
        torch::Tensor distances = TensorOperations::pairwise_distances(positions);
        torch::Tensor expected_distances = torch::tensor({{0.0f, 1.0f, 1.0f},
                                                         {1.0f, 0.0f, std::sqrt(2.0f)},
                                                         {1.0f, std::sqrt(2.0f), 0.0f}});
        assert_tensor_close(distances, expected_distances, tolerance, "pairwise distances");
    }
    
    void test_numerical_stability() {
        std::cout << "\n=== Testing Numerical Stability ===" << std::endl;
        
        DNA2Parameters params;
        params.initialize_tensors();
        
        // Test very small values
        torch::Tensor tiny = torch::tensor(1e-10f);
        torch::Tensor f1_tiny = MathematicalFunctions::f1(tiny, torch::tensor(0.4f), torch::tensor(8.0f));
        assert_almost_equal(1.0f, f1_tiny.item<float>(), 1e-3f, "f1 with tiny input");
        
        // Test very large values
        torch::Tensor large = torch::tensor(1e10f);
        torch::Tensor f1_large = MathematicalFunctions::f1(large, torch::tensor(0.4f), torch::tensor(8.0f));
        assert_almost_equal(0.0f, f1_large.item<float>(), tolerance, "f1 with large input");
        
        // Test normalization of very small vectors
        torch::Tensor tiny_vec = torch::tensor({1e-10f, 1e-10f, 1e-10f});
        torch::Tensor norm_tiny = TensorOperations::normalize(tiny_vec);
        assert_tensor_close(norm_tiny, torch::tensor({1.0f, 0.0f, 0.0f}), 1e-3f, 
                          "normalize tiny vector");
        
        // Test angle with nearly parallel vectors
        torch::Tensor v_parallel1 = torch::tensor({1.0f, 0.0f, 0.0f});
        torch::Tensor v_parallel2 = torch::tensor({1.0f, 1e-10f, 0.0f});
        torch::Tensor angle_parallel = TensorOperations::angle_between(v_parallel1, v_parallel2);
        assert_almost_equal(0.0f, angle_parallel.item<float>(), 1e-6f, "angle nearly parallel");
    }
    
    void test_derivative_consistency() {
        std::cout << "\n=== Testing Derivative Consistency ===" << std::endl;
        
        DNA2Parameters params;
        params.initialize_tensors();
        
        float h = 1e-6f;  // Small step for numerical derivative
        
        // Test f1 derivative consistency
        for (float r = 0.1f; r < 0.4f; r += 0.1f) {
            torch::Tensor r_tensor = torch::tensor(r);
            torch::Tensor r_plus = torch::tensor(r + h);
            torch::Tensor r_minus = torch::tensor(r - h);
            
            torch::Tensor f1_plus = MathematicalFunctions::f1(r_plus, torch::tensor(0.4f), torch::tensor(8.0f));
            torch::Tensor f1_minus = MathematicalFunctions::f1(r_minus, torch::tensor(0.4f), torch::tensor(8.0f));
            float numerical_deriv = (f1_plus.item<float>() - f1_minus.item<float>()) / (2.0f * h);
            
            torch::Tensor analytical_deriv = MathematicalFunctions::f1_derivative(r_tensor, torch::tensor(0.4f), torch::tensor(8.0f));
            
            float error = std::abs(numerical_deriv - analytical_deriv.item<float>());
            assert_almost_equal(0.0f, error, 1e-4f, "f1 derivative consistency at r=" + std::to_string(r));
        }
        
        // Test f4 derivative consistency
        for (float theta = 0.1f; theta < M_PI; theta += 0.5f) {
            torch::Tensor theta_tensor = torch::tensor(theta);
            torch::Tensor theta_plus = torch::tensor(theta + h);
            torch::Tensor theta_minus = torch::tensor(theta - h);
            
            torch::Tensor f4_plus = MathematicalFunctions::f4(theta_plus, torch::tensor(M_PI/2), torch::tensor(6.0f));
            torch::Tensor f4_minus = MathematicalFunctions::f4(theta_minus, torch::tensor(M_PI/2), torch::tensor(6.0f));
            float numerical_deriv = (f4_plus.item<float>() - f4_minus.item<float>()) / (2.0f * h);
            
            torch::Tensor analytical_deriv = MathematicalFunctions::f4_derivative(theta_tensor, torch::tensor(M_PI/2), torch::tensor(6.0f));
            
            float error = std::abs(numerical_deriv - analytical_deriv.item<float>());
            assert_almost_equal(0.0f, error, 1e-4f, "f4 derivative consistency at theta=" + std::to_string(theta));
        }
    }
    
    void run_all_tests() {
        std::cout << "DNA2 Mathematical Functions Test Suite" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        test_f1_function();
        test_f2_function();
        test_f4_function();
        test_f5_function();
        test_tensor_operations();
        test_geometric_functions();
        test_numerical_stability();
        test_derivative_consistency();
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed_tests << "/" << total_tests << " tests" << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << "✓ All mathematical function tests passed!" << std::endl;
        } else {
            std::cout << "✗ Some tests failed!" << std::endl;
        }
    }
};

int main() {
    try {
        TestMathematicalFunctions tester;
        tester.run_all_tests();
        return (tester.passed_tests == tester.total_tests) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}