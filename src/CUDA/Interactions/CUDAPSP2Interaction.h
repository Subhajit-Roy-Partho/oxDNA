#ifndef CUDAPSP2INTERACTION_H_
#define CUDAPSP2INTERACTION_H_

#include "CUDABaseInteraction.h"
#include "../cuda_utils/CUDA_lr_common.cuh"
#include "../Lists/CUDASimpleVerletList.h"
#include "../Lists/CUDANoList.h"
#include "../../Interactions/PSP2Interaction.h"

class CUDAPSP2Interaction: public CUDABaseInteraction, public PSP2Interaction {
public:
    CUDAPSP2Interaction();
    virtual ~CUDAPSP2Interaction();
    void get_settings(input_file &inp) override;
    void cuda_init(int N) override;
    c_number get_cuda_rcut() {
        return this->get_rcut();
    }
    void compute_forces(CUDABaseList *lists,c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torque, LR_bonds *d_bonds, CUDABox *d_box);

};


#endif