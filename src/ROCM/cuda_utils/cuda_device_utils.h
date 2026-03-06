/*
 * cuda_device_info.h
 *
 *  Created on: 30/lug/2009
 *      Author: lorenzo
 */

#ifndef CUDA_DEVICE_INFO_H_
#define CUDA_DEVICE_INFO_H_

#include <hip/hip_runtime.h>

int get_device_count();
void check_device_existance(int device);
hipDeviceProp_t get_current_device_prop();
hipDeviceProp_t get_device_prop(int device);

#endif /* CUDA_DEVICE_INFO_H_ */
