#ifndef ROCM_CURAND_KERNEL_WRAPPER_H_
#define ROCM_CURAND_KERNEL_WRAPPER_H_

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

typedef struct {
	unsigned long long state;
	bool has_spare;
	float spare;
	double spare_double;
} curandState;

using hiprandState = curandState;

__device__ inline unsigned long long _curand_step(curandState *state) {
	unsigned long long x = state->state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	state->state = x;
	return x * 2685821657736338717ULL;
}

__device__ inline void curand_init(unsigned long long seed, unsigned long long subsequence,
		unsigned long long offset, curandState *state) {
	unsigned long long mix = seed ^ (0x9E3779B97F4A7C15ULL * (subsequence + 1ULL));
	mix += 0xBF58476D1CE4E5B9ULL * offset;
	if(mix == 0ULL) {
		mix = 0xA24BAED4963EE407ULL;
	}
	state->state = mix;
	state->has_spare = false;
	state->spare = 0.f;
	state->spare_double = 0.;
}

__device__ inline float curand_uniform(curandState *state) {
	constexpr float inv = 1.0f / 18446744073709551616.0f;
	float value = static_cast<float>(_curand_step(state) * inv);
	return (value > 0.f) ? value : 5.9604645e-8f;
}

__device__ inline float curand_normal(curandState *state) {
	if(state->has_spare) {
		state->has_spare = false;
		return state->spare;
	}

	float u1 = curand_uniform(state);
	float u2 = curand_uniform(state);
	float radius = sqrtf(-2.f * logf(u1));
	float angle = 6.2831853071795864769f * u2;
	state->spare = radius * sinf(angle);
	state->has_spare = true;
	return radius * cosf(angle);
}

__device__ inline double curand_normal_double(curandState *state) {
	if(state->has_spare) {
		state->has_spare = false;
		return state->spare_double;
	}

	double u1 = static_cast<double>(curand_uniform(state));
	double u2 = static_cast<double>(curand_uniform(state));
	double radius = sqrt(-2.0 * log(u1));
	double angle = 6.2831853071795864769 * u2;
	state->spare_double = radius * sin(angle);
	state->has_spare = true;
	return radius * cos(angle);
}

#define hiprand_uniform curand_uniform
#define hiprand_init curand_init

#endif /* ROCM_CURAND_KERNEL_WRAPPER_H_ */
