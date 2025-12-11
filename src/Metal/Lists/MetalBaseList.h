/**
 * @file    MetalBaseList.h
 * @date    Created for Metal backend
 * @author  antigravity
 *
 *
 */

#ifndef METALBASELIST_H_
#define METALBASELIST_H_

#include "../MetalUtils.h"
#include "../metal_utils/MetalBox.h"
#include "../metal_defs.h"

/**
 * @brief Abstract class for list-based force computing on Metal
 *
 * Child classes have to implement methods to read the input file, update the lists
 * and clean up at the end of the simulation
 */

class MetalBaseList {
protected:
	bool _use_edge;
	int _N;
	MetalBox *_h_metal_box;
    // We keep a reference to the buffer containing box data on GPU
    id<MTLBuffer> _d_metal_box;

public:
    // Using raw id<MTLBuffer> for flexibility or MetalBuffer wrapper?
    // CUDABaseList used raw pointers. MetalBuffer is checking errors.
    // Let's use MetalBuffer wrapper or just raw pointers managed manually if needed.
    // But MetalBuffer is defined in metal_defs.h.
    // However, CUDABaseList exposed raw pointers publically.
    // Let's use MetalBuffer helper but expose the internal buffer if needed?
    // Or just use raw id<MTLBuffer> and manage them.
    
    // For consistency with CUDA which uses raw pointers (which are device pointers),
    // we can use id<MTLBuffer>.
    
	id<MTLBuffer> d_matrix_neighs = nil;
	id<MTLBuffer> d_number_neighs = nil;
	id<MTLBuffer> d_edge_list = nil;
	int N_edges = 0;

	MetalBaseList() :
					_use_edge(false),
					_N(-1),
					_h_metal_box(nullptr),
					_d_metal_box(nil) {
	}
	;
	virtual ~MetalBaseList() {
	}
	;

	virtual void get_settings(input_file &inp) = 0;
	virtual void metal_init(int N, m_number rcut, MetalBox *h_metal_box, id<MTLBuffer> d_metal_box, id<MTLDevice> device, id<MTLLibrary> library) {
		_h_metal_box = h_metal_box;
		_d_metal_box = d_metal_box;
		_N = N;
	}

    // Update requires Metal buffers
	virtual void update(id<MTLBuffer> poss, id<MTLBuffer> list_poss, id<MTLBuffer> bonds) = 0;

	bool use_edge() {
		return _use_edge;
	}

	virtual void clean() = 0;
};

#endif /* METALBASELIST_H_ */
