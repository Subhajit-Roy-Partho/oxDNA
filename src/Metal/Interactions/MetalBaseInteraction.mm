/*
 * MetalBaseInteraction.mm
 *
 *  Created for Metal backend
 */

#include "MetalBaseInteraction.h"
#include "../../Utilities/oxDNAException.h"
#include "../../Utilities/ConfigInfo.h"

MetalBaseInteraction::MetalBaseInteraction() {

}

MetalBaseInteraction::~MetalBaseInteraction() {
	if(_use_edge) {
		// Released by ARC/nil
        _d_edge_forces = nil;
        _d_edge_torques = nil;
	}
    _d_st = nil;
}

void MetalBaseInteraction::get_metal_settings(input_file &inp) {
	int update_st_every = 0;
	getInputInt(&inp, "Metal_update_stress_tensor_every", &update_st_every, 0);
	if(update_st_every > 0) {
		_update_st = true;
	}

	getInputBool(&inp, "use_edge", &_use_edge, 0);
	if(_use_edge) {
		if(!_edge_compatible) {
			throw oxDNAException("The selected Metal interaction is not compatible with 'use_edge = true'");
		}

		getInputInt(&inp, "edge_n_forces", &_n_forces, 0);
		if(_n_forces < 1) {
			throw oxDNAException("edge_n_forces must be > 0");
		}
	}
}

void MetalBaseInteraction::metal_init(int N, id<MTLDevice> device, id<MTLLibrary> library) {
	_N = N;
    _device = device;
    _library = library;

	if(_use_edge && _d_edge_forces == nil) {
		_d_edge_forces = MetalUtils::allocate_buffer<m_number4>(_device, _N * _n_forces, MTLResourceStorageModePrivate);
        _d_edge_torques = MetalUtils::allocate_buffer<m_number4>(_device, _N * _n_forces, MTLResourceStorageModePrivate);
        
        // Zero out buffers
        id<MTLCommandQueue> queue = [_device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder fillBuffer:_d_edge_forces range:NSMakeRange(0, _N * _n_forces * sizeof(m_number4)) value:0];
        [blitEncoder fillBuffer:_d_edge_torques range:NSMakeRange(0, _N * _n_forces * sizeof(m_number4)) value:0];
        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
	}

	if(_update_st) {
        _d_st = MetalUtils::allocate_buffer<MetalStressTensor>(_device, N, MTLResourceStorageModePrivate);
	}
}

void MetalBaseInteraction::_sum_edge_forces(id<MTLBuffer> d_forces) {
    // Implement sum kernel dispatch
    // Need PSO for sum_edge_forces
    // For now throwing exception or implementing empty
    // Revisit when implementing CUDABaseInteraction equivalents fully
}

void MetalBaseInteraction::_sum_edge_forces_torques(id<MTLBuffer> d_forces, id<MTLBuffer> d_torques) {
    // Implement sum kernel dispatch
}
