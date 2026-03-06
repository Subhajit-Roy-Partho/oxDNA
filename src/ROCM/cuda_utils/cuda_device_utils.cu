#include "cuda_device_utils.h"

#include <cstdlib>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

int get_device_count() {
	int deviceCount = 0;
	if(hipGetDeviceCount(&deviceCount) != hipSuccess) {
		fprintf(stderr, "hipGetDeviceCount FAILED, CUDA Driver and Runtime CUDA Driver and Runtime version may be mismatched, exiting.\n");
		exit(1);
	}

	return deviceCount;
}

void check_device_existance(int device) {
	if(device >= get_device_count()) {
		fprintf(stderr, "The selected device doesn't exist, exiting.\n");
		exit(1);
	}
}

hipDeviceProp_t get_current_device_prop() {
	int curr_dev;
	hipGetDevice(&curr_dev);
	return get_device_prop(curr_dev);
}

hipDeviceProp_t get_device_prop(int device) {
	check_device_existance(device);

	hipDeviceProp_t deviceProp;
	hipGetDeviceProperties(&deviceProp, device);

	return deviceProp;
}
