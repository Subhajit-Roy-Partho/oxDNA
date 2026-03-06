/*
 * Minimal HIP-compatible replacement for the legacy CUDA helper header.
 */

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

template<typename T>
inline void check(T result, const char *const func, const char *const file, int const line) {
	if(result != hipSuccess) {
		fprintf(stderr, "%s(%d): HIP error %d at %s: %s\n",
				file, line, static_cast<int>(result), func, hipGetErrorString(result));
		std::exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
	hipError_t err = hipGetLastError();
	if(hipSuccess != err) {
		fprintf(stderr, "%s(%d): getLastCudaError() HIP error: %s: (%d) %s.\n",
				file, line, errorMessage, static_cast<int>(err), hipGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

#define getLastCudaError(msg) __getLastCudaError((msg), __FILE__, __LINE__)

#endif /* HELPER_CUDA_H */
