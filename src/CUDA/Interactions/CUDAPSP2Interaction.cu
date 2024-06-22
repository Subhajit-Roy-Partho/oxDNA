#include "CUDAPSP2Interaction.h"

__constant__ int MD_N[1];
__constant__ float spring[MAXPatches][5];
__constant__ float patch[MAXSprings][5];


__device__ int strand[MAXparticles];
__device__ int connection[MAXparticles][MAXneighbour+1];
__device__ int springParticle[MAXparticles][MAXSpringPerParticle];
__device__ int invSpringParticle[MAXparticles][MAXSpringPerParticle];
__device__ int patchParticle[MAXparticles][MAXPatchPerParticle];


__device__ float radius[MAXparticles];


__device__ void TorqueSpring(int p, int q,int i,c_number4 &a1,c_number4 &a2,c_number4 &a3,c_number4 &b1,c_number4 &b2,c_number4 &b3,c_number4 &r, c_number4 &F, c_number4 &T){
    c_number k,r0;
    k= spring[springParticle[p][i]][0];
    r0 = spring[springParticle[p][i]][1];

    c_number4 rP={
        a1.x*spring[springParticle[p][i]][2]+a2.x*spring[springParticle[p][i]][3]+a3.x*spring[springParticle[p][i]][4],
        a1.y*spring[springParticle[p][i]][2]+a2.y*spring[springParticle[p][i]][3]+a3.y*spring[springParticle[p][i]][4],
        a1.z*spring[springParticle[p][i]][2]+a2.z*spring[springParticle[p][i]][3]+a3.z*spring[springParticle[p][i]][4],
        0.0
    };
    c_number4 rQ={
        b1.x*spring[invSpringParticle[p][i]][2]+b2.x*spring[invSpringParticle[p][i]][3]+b3.x*spring[invSpringParticle[p][i]][4],
        b1.y*spring[invSpringParticle[p][i]][2]+b2.y*spring[invSpringParticle[p][i]][3]+b3.y*spring[invSpringParticle[p][i]][4],
        b1.z*spring[invSpringParticle[p][i]][2]+b2.z*spring[invSpringParticle[p][i]][3]+b3.z*spring[invSpringParticle[p][i]][4],
        0.0
    };
    c_number4 distr={
        r.x-rP.x+rQ.x,
        r.y-rP.y+rQ.y,
        r.z-rP.z+rQ.z,
        0.0
    };
    c_number dist =sqrtf(CUDA_DOT(distr,distr));
    c_number effDist = dist-r0;
    F.w+= 0.25*k*effDist*effDist;

    c_number4 force = distr*(k*effDist/dist);
    F.x += force.x;
    F.y += force.y;
    F.z += force.z;
}

__global__ void CUDAPSP2Particle(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, int *matrix_neighs, int *number_neighs, CUDABox *box){
    if(IND >= MD_N[0]) return; // for i > N leave
    c_number4 F = forces[IND];
    c_number4 T = torques[IND];
	c_number4 ppos = poss[IND];
    c_number4 a1, a2, a3;
    get_vectors_from_quat(orientations[IND], a1, a2, a3);

    for(int p=0;p<connection[IND][0];p++){
        int id = connection[IND][p+1];
        c_number4 b1,b2,b3;
        get_vectors_from_quat(orientations[id], b1, b2, b3);
        c_number4 r = box->minimum_image(ppos, poss[id]);
    }

    int num_neighs = NUMBER_NEIGHBOURS(IND, number_neighs);
    for(int j = 0; j < num_neighs; j++) {
		int k_index = NEXT_NEIGHBOUR(IND, j, matrix_neighs);
    }
    forces[IND] = F;
	torques[IND] = _vectors_transpose_c_number4_product(a1, a2, a3, T);

};

CUDAPSP2Interaction::CUDAPSP2Interaction():PSP2Interaction() {
    // Constructor
}

CUDAPSP2Interaction::~CUDAPSP2Interaction() {
    // Destructor
}

void CUDAPSP2Interaction::get_settings(input_file &inp) {
    // Get the settings
    PSP2Interaction::get_settings(inp);
}

void CUDAPSP2Interaction::cuda_init(int N){
    // Initialize the interaction
    CUDABaseInteraction::cuda_init(N);
    PSP2Interaction::init();
    std::vector<BaseParticle *> particles(N);
    PSP2Interaction::allocate_particles(particles);
    int N_strands;
    PSP2Interaction::read_topology(&N_strands,particles);
    for(int i = 0; i < N; i++) {
        delete particles[i];
    }

    // Copy the data to the device
    // Integers
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(strand, &particleStrand, sizeof(int)*MAXparticles));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(connection, &connections, sizeof(int)*MAXparticles*(MAXneighbour+1)));

    // Floats
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(radius, &particleRadius, sizeof(float)*MAXparticles));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(spring, &Springs, sizeof(float)*MAXSprings*5));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(patch, &Patches, sizeof(float)*MAXPatches*5));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(springParticle, &ParticleSprings, sizeof(int)*MAXparticles*MAXSpringPerParticle));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(invSpringParticle, &invParticleSprings, sizeof(int)*MAXparticles*MAXSpringPerParticle));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(patchParticle, &ParticlePatches, sizeof(int)*MAXparticles*MAXPatchPerParticle));


}

void CUDAPSP2Interaction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
    // Compute the forces per particles
    CUDAPSP2Particle <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>> 
    (d_poss, d_orientations, d_forces,  d_torques, lists->d_matrix_neighs, lists->d_number_neighs, d_box);
    CUT_CHECK_ERROR("Kernel failed, something quite exciting");
}