/**
 * @file    CUDACellsList.h
 * @brief   Zero-skin, rebuild-every-step CUDA neighbour list.
 *
 * CUDACellsList is a CUDASimpleVerletList run with verlet_skin = 0, so the
 * list build radius equals the interaction cutoff and the list is rebuilt
 * from the cell assignment every step. Unlike CUDABinVerletList it supports
 * orthorhombic (non-cubic) boxes and the edge-based approach, since it
 * inherits init/update/clean unchanged from CUDASimpleVerletList.
 */
#ifndef CUDACELLSLIST_H_
#define CUDACELLSLIST_H_

#include "CUDASimpleVerletList.h"

class CUDACellsList: public CUDASimpleVerletList {
public:
	CUDACellsList();
	virtual ~CUDACellsList();

	void get_settings(input_file &inp);
};

#endif /* CUDACELLSLIST_H_ */
