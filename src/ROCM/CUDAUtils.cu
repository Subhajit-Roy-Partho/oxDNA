#include "hip/hip_runtime.h"
/*
 * GpuUtils.cpp
 *
 *  Created on: 24/set/2010
 *      Author: lorenzo
 */

#include "CUDAUtils.h"
#include <algorithm>
#include <vector>

size_t GpuUtils::_allocated_dev_mem = 0;

namespace {

constexpr int REDUCTION_THREADS = 256;

template<typename VecType>
__device__ inline double kinetic_energy_contribution(const VecType &v, const VecType &L, bool any_rigid_body) {
	double translational = static_cast<double>(v.x) * static_cast<double>(v.x)
		+ static_cast<double>(v.y) * static_cast<double>(v.y)
		+ static_cast<double>(v.z) * static_cast<double>(v.z);
	double rotational = 0.;
	if(any_rigid_body) {
		rotational = static_cast<double>(L.x) * static_cast<double>(L.x)
			+ static_cast<double>(L.y) * static_cast<double>(L.y)
			+ static_cast<double>(L.z) * static_cast<double>(L.z);
	}
	return 0.5 * (translational + rotational);
}

__global__ void reduce_c_number4_w_kernel(const c_number4 *values, int N, double *partial_sums) {
	__shared__ double shared_sums[REDUCTION_THREADS];

	double local_sum = 0.;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
		local_sum += static_cast<double>(values[idx].w);
	}
	shared_sums[threadIdx.x] = local_sum;
	__syncthreads();

	for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if(threadIdx.x < stride) {
			shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		partial_sums[blockIdx.x] = shared_sums[0];
	}
}

template<typename VecType>
__global__ void reduce_kinetic_energy_kernel(const VecType *vels, const VecType *Ls, int N, bool any_rigid_body, double *partial_sums) {
	__shared__ double shared_sums[REDUCTION_THREADS];

	double local_sum = 0.;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
		local_sum += kinetic_energy_contribution(vels[idx], Ls[idx], any_rigid_body);
	}
	shared_sums[threadIdx.x] = local_sum;
	__syncthreads();

	for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if(threadIdx.x < stride) {
			shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		partial_sums[blockIdx.x] = shared_sums[0];
	}
}

double reduce_partial_sums(double *device_partial_sums, int blocks) {
	std::vector<double> host_partial_sums(blocks);
	CUDA_SAFE_CALL(hipMemcpy(host_partial_sums.data(), device_partial_sums, sizeof(double) * blocks, hipMemcpyDeviceToHost));

	double total = 0.;
	for(double value : host_partial_sums) {
		total += value;
	}
	return total;
}

double launch_c_number4_w_reduction(const c_number4 *values, int N) {
	if(N <= 0) {
		return 0.;
	}

	const int blocks = std::max(1, std::min((N + REDUCTION_THREADS - 1) / REDUCTION_THREADS, 1024));
	double *device_partial_sums = nullptr;
	CUDA_SAFE_CALL(hipMalloc(&device_partial_sums, sizeof(double) * blocks));

	reduce_c_number4_w_kernel<<<blocks, REDUCTION_THREADS>>>(values, N, device_partial_sums);
	CUT_CHECK_ERROR("double reduction kernel error");

	double total = reduce_partial_sums(device_partial_sums, blocks);
	CUDA_SAFE_CALL(hipFree(device_partial_sums));
	return total;
}

template<typename VecType>
double launch_kinetic_energy_reduction(const VecType *vels, const VecType *Ls, int N, bool any_rigid_body) {
	if(N <= 0) {
		return 0.;
	}

	const int blocks = std::max(1, std::min((N + REDUCTION_THREADS - 1) / REDUCTION_THREADS, 1024));
	double *device_partial_sums = nullptr;
	CUDA_SAFE_CALL(hipMalloc(&device_partial_sums, sizeof(double) * blocks));

	reduce_kinetic_energy_kernel<VecType><<<blocks, REDUCTION_THREADS>>>(vels, Ls, N, any_rigid_body, device_partial_sums);
	CUT_CHECK_ERROR("kinetic energy reduction kernel error");

	double total = reduce_partial_sums(device_partial_sums, blocks);
	CUDA_SAFE_CALL(hipFree(device_partial_sums));
	return total;
}

}

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
	return sum_c_number4_w_to_double_on_GPU(dv, N);
}

double GpuUtils::sum_c_number4_w_to_double_on_GPU(c_number4 *dv, int N) {
	return launch_c_number4_w_reduction(dv, N);
}

double GpuUtils::sum_kinetic_energy_on_GPU(c_number4 *vels, c_number4 *Ls, int N, bool any_rigid_body) {
	return launch_kinetic_energy_reduction(vels, Ls, N, any_rigid_body);
}

double GpuUtils::sum_kinetic_energy_on_GPU(LR_double4 *vels, LR_double4 *Ls, int N, bool any_rigid_body) {
	return launch_kinetic_energy_reduction(vels, Ls, N, any_rigid_body);
}
