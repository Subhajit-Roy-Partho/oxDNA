/*
 * CUDACellsList.cu
 *
 * Zero-skin, rebuild-every-step CUDA neighbour list. Inherits init/update/clean
 * unchanged from CUDASimpleVerletList; only get_settings differs (verlet_skin
 * must be 0 or unset and is forced to 0).
 */

#include "CUDACellsList.h"
#include "../../Utilities/oxDNAException.h"

CUDACellsList::CUDACellsList() {

}

CUDACellsList::~CUDACellsList() {

}

void CUDACellsList::get_settings(input_file &inp) {
	getInputBool(&inp, "cells_auto_optimisation", &_auto_optimisation, 0);
	getInputBool(&inp, "print_problematic_ids", &_print_problematic_ids, 0);
	getInputNumber(&inp, "max_density_multiplier", &_max_density_multiplier, 0);
	getInputBool(&inp, "use_edge", &_use_edge, 0);
	if(_use_edge) {
		OX_LOG(Logger::LOG_INFO, "Using edge-based approach");
	}

	// verlet_skin is optional here (unlike in the base class, where it is
	// mandatory): if present and non-zero, abort. A non-zero skin would make
	// the list build radius larger than rcut while the backend's staleness
	// criterion (particle displacement vs skin in the first_step kernel)
	// would allow steps without a rebuild, silently missing pairs. With skin
	// 0 the backend rebuilds the list every step automatically.
	c_number skin = 0.;
	if(getInputNumber(&inp, "verlet_skin", &skin, 0) == KEY_FOUND && skin != 0.) {
		throw oxDNAException("CUDA_list = cells requires verlet_skin = 0 or unset (found %g). The cells list uses a zero skin and is rebuilt every step.", (double) skin);
	}
	_verlet_skin = 0.;
}
