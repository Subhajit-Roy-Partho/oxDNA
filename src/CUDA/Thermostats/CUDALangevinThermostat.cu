/*
 * CUDALangevinThermostat.cu
 *
 *  Created on: Sep 1, 2014
 *      Author: mzimmer
 */

#include "CUDALangevinThermostat.h"

#include "../../Utilities/ConfigInfo.h"

#include <curand_kernel.h>

__global__ void langevin_thermostat(c_number *invmass , c_number *invmr2,curandState *rand_state, c_number4 *vels, c_number4 *Ls, c_number _dt, c_number gamma_trans, c_number rescale_factor_trans, c_number gamma_rot, c_number rescale_factor_rot, int N) {
	if(IND < N) {
		curandState state = rand_state[IND];
		c_number f_vx, f_vy, f_vz;
		c_number f_Lx, f_Ly, f_Lz;
		gaussian(state, f_vx, f_vy);
		gaussian(state, f_vz, f_Lx);
		gaussian(state, f_Ly, f_Lz);
		rand_state[IND] = state;

		c_number4 v = vels[IND];
		c_number4 L = Ls[IND];
		c_number vfactor = sqrtf(invmass[IND]);
		c_number Lfactor = sqrtf(invmr2[IND]); 

		//Could define operators for GPU_quat
		v.x += _dt * (-gamma_trans * v.x + f_vx * rescale_factor_trans*vfactor);
		v.y += _dt * (-gamma_trans * v.y + f_vy * rescale_factor_trans*vfactor);
		v.z += _dt * (-gamma_trans * v.z + f_vz * rescale_factor_trans*vfactor);
		L.x += _dt * (-gamma_rot * L.x + f_Lx * rescale_factor_rot*Lfactor);
		L.y += _dt * (-gamma_rot * L.y + f_Ly * rescale_factor_rot*Lfactor);
		L.z += _dt * (-gamma_rot * L.z + f_Lz * rescale_factor_rot*Lfactor);

		//if (IND==5) printf("Diff is: %f %f %f \n",L.x-Ls[IND].x, L.y-Ls[IND].y, L.z-Ls[IND].z);

		vels[IND] = v;
		Ls[IND] = L;
	}
}

CUDALangevinThermostat::CUDALangevinThermostat() :
				CUDABaseThermostat(),
				LangevinThermostat() {

}

CUDALangevinThermostat::~CUDALangevinThermostat() {

}

void CUDALangevinThermostat::get_settings(input_file &inp) {
	LangevinThermostat::get_settings(inp);
	CUDABaseThermostat::get_cuda_settings(inp);
}

void CUDALangevinThermostat::init() {
	LangevinThermostat::init();

	this->_setup_rand(CONFIG_INFO->N());
}

bool CUDALangevinThermostat::would_activate(llint curr_step) {
	return 1;
}

void CUDALangevinThermostat::apply_cuda(c_number *d_invmass,c_number *d_invmr2,c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
	if(!would_activate(curr_step)) return;

	langevin_thermostat
		<<<this->_launch_cfg.blocks, this->_launch_cfg.threads_per_block>>>
		(d_invmass,d_invmr2,this->_d_rand_state, d_vels, d_Ls, this->_dt, this->_gamma_trans, this-> _rescale_factor_trans, this->_gamma_rot, this->_rescale_factor_rot, CONFIG_INFO->N());
}
