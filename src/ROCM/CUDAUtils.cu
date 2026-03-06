#include "hip/hip_runtime.h"
/*
 * GpuUtils.cpp
 *
 *  Created on: 24/set/2010
 *      Author: lorenzo
 */

#include "CUDAUtils.h"
#include <vector>

size_t GpuUtils::_allocated_dev_mem = 0;

__global__ void print_array(int *v, int N) {
	for(int i = 0; i < N; i++)
		printf("%d %d\n", i, v[i]);
}

__global__ void print_array(float *v, int N) {
	for(int i = 0; i < N; i++)
		printf("%d %f\n", i, v[i]);
}

__global__ void print_array(double *v, int N) {
	for(int i = 0; i < N; i++)
		printf("%d %lf\n", i, v[i]);
}

__global__ void print_array(LR_double4 *v, int N) {
	for(int i = 0; i < N; i++)
		printf("%d %lf %lf %lf %lf\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
}

__global__ void print_array(float4 *v, int N) {
	for(int i = 0; i < N; i++)
		printf("%d %lf %lf %lf %lf\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
}

template<typename T>
__global__ void check_thresold(T *v, int N, int t) {
	for(int i = 0; i < N; i++)
		if(v[i] >= t) printf("%d %d\n", i, v[i]);
}

template<typename T>
void GpuUtils::print_device_array(T *v, int N) {
	print_array
		<<<1,1>>>
		(v, N);
		CUT_CHECK_ERROR("print_device_array error");
	(void) hipDeviceSynchronize();
}

template void GpuUtils::print_device_array<float4>(float4 *, int);

template<typename T>
void GpuUtils::check_device_thresold(T *v, int N, int t) {
check_thresold<T>
		<<<1,1>>>
		(v, N, t);
		CUT_CHECK_ERROR("check_device_thresold error");
	(void) hipDeviceSynchronize();
}

c_number4 GpuUtils::sum_c_number4_on_GPU(c_number4 *dv, int N) {
	std::vector<c_number4> host_values(N);
	CUDA_SAFE_CALL(hipMemcpy(host_values.data(), dv, sizeof(c_number4) * N, hipMemcpyDeviceToHost));
	c_number4 sum = { 0., 0., 0., 0. };
	for(const auto &value : host_values) {
		sum.x += value.x;
		sum.y += value.y;
		sum.z += value.z;
		sum.w += value.w;
	}
	return sum;
}

double GpuUtils::sum_c_number4_to_double_on_GPU(c_number4 *dv, int N) {
	std::vector<c_number4> host_values(N);
	CUDA_SAFE_CALL(hipMemcpy(host_values.data(), dv, sizeof(c_number4) * N, hipMemcpyDeviceToHost));
	double sum = 0.;
	for(const auto &value : host_values) {
		sum += static_cast<double>(value.w);
	}
	return sum;
}
