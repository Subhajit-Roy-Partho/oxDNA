/*
 * CUDABrownianThermostat.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: rovigatti
 */

#include "CUDABrownianThermostat.h"

#include "../cuda_utils/CUDA_lr_common.cuh"
#include "../../Utilities/ConfigInfo.h"

#include <curand_kernel.h>

__global__ void brownian_thermostat(c_number *invmass,c_number *invmr2,curandState *rand_state, c_number4 *vels, c_number4 *Ls, c_number rescale_factor, c_number pt, c_number pr, int N) {
	if(IND < N) {
		curandState state = rand_state[IND];

		if(curand_uniform(&state) < pt) {
			c_number4 v;
			c_number trash;

			gaussian(state, v.x, v.y);
			gaussian(state, v.z, trash);

			c_number rescale = rescale_factor*sqrtf(invmass[IND]);

			v.x *= rescale;
			v.y *= rescale;
			v.z *= rescale;
			v.w = (v.x * v.x + v.y * v.y + v.z * v.z) * (c_number) 0.5f;

			vels[IND] = v;
		}

		if(curand_uniform(&state) < pr) {
			c_number4 L;
			c_number trash;

			gaussian(state, L.x, L.y);
			gaussian(state, L.z, trash);

			c_number rescale = rescale_factor*sqrtf(invmr2[IND]);

			L.x *= rescale;
			L.y *= rescale;
			L.z *= rescale;
			L.w = (L.x * L.x + L.y * L.y + L.z * L.z) * (c_number) 0.5f;

			Ls[IND] = L;
		}

		rand_state[IND] = state;
	}
}

CUDABrownianThermostat::CUDABrownianThermostat() :
				CUDABaseThermostat(),
				BrownianThermostat() {

}

CUDABrownianThermostat::~CUDABrownianThermostat() {

}

void CUDABrownianThermostat::get_settings(input_file &inp) {
	BrownianThermostat::get_settings(inp);
	CUDABaseThermostat::get_cuda_settings(inp);
}

void CUDABrownianThermostat::init() {
	BrownianThermostat::init();

	this->_setup_rand(CONFIG_INFO->N());
}

bool CUDABrownianThermostat::would_activate(llint curr_step) {
	return (curr_step % this->_newtonian_steps == 0);
}

void CUDABrownianThermostat::apply_cuda(c_number *d_invmass,c_number *d_invmr2,c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
	if(!would_activate(curr_step)) return;

	brownian_thermostat
		<<<this->_launch_cfg.blocks, this->_launch_cfg.threads_per_block>>>
		(d_invmass,d_invmr2,this->_d_rand_state, d_vels, d_Ls, this->_rescale_factor, this->_pt, this->_pr, CONFIG_INFO->N());
}
