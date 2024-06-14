/*
 * CUDANoThermostat.cu
 *
 *  Created on: Feb 15, 2013
 *      Author: rovigatti
 */

#include "CUDANoThermostat.h"

CUDANoThermostat::CUDANoThermostat() :
				NoThermostat() {

}

CUDANoThermostat::~CUDANoThermostat() {

}

void CUDANoThermostat::apply_cuda(c_number *d_invmass, c_number *d_invmr2, c_number4 *d_poss, GPU_quat *d_orientationss, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
	return;
}
