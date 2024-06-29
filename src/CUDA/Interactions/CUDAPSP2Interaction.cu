#include "CUDAPSP2Interaction.h"

__constant__ int MD_N[1];
__constant__ float spring[MAXPatches][5];
__constant__ float patch[MAXSprings][5];
__constant__ float sigma;
__constant__ float Rstar;
__constant__ float patchyb;
__constant__ float Rc;
__constant__ float patchyCut2;
__constant__ float patchyCut;
__constant__ float alpha;
__constant__ float alphaB2;


__device__ int strand[MAXparticles];
__device__ int connection[MAXparticles][MAXneighbour+1];
__device__ int springParticle[MAXparticles][MAXSpringPerParticle];
__device__ int invSpringParticle[MAXparticles][MAXSpringPerParticle];
__device__ int patchParticle[MAXparticles][MAXPatchPerParticle+1];


__device__ float radius[MAXparticles];


__forceinline__ __device__ void TorqueSpring(const int &p, const int &q, const int &i, const c_number4 &a1,const c_number4 &a2,const c_number4 &a3,const c_number4 &b1,const c_number4 &b2,const c_number4 &b3,const c_number4 &r, c_number4 &F, c_number4 &T){
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
    c_number4 torque = _cross(rP,force);
    F.x += force.x;
    F.y += force.y;
    F.z += force.z;
    T.x += torque.x;
    T.y += torque.y;
    T.z += torque.z;
}

__forceinline__ __device__ void repulsiveLinear(const c_number4 &r,const c_number &r2, const c_number &rmod,const c_number &totRad,c_number4 &F){
    c_number rSigma = totRad*sigma;
    c_number rRstar = totRad*Rstar;
    c_number rRc = totRad*Rc;
    c_number rB = patchyb/(totRad*totRad);
    if(r2<rRc*rRc){
        if(r2>rRstar*rRstar){
            c_number rrc = rmod - rRc;
            c_number fmod = 2*rB*rrc/rmod;
            F.x += fmod*r.x;
            F.y += fmod*r.y;
            F.z += fmod*r.z;
            F.w += rB*SQR(rrc);
        }else{
            c_number lj_part = CUB(SQR(rSigma)/r2);
            c_number fmod = 24*(lj_part-2*SQR(lj_part))/r2;
            F.x += fmod*r.x;
            F.y += fmod*r.y;
            F.z += fmod*r.z;
            F.w += 4*(SQR(lj_part)-lj_part);
        }
    }
}

__forceinline__ __device__ void simplePatch(const int &p, const int &q, const c_number4 &a1,const c_number4 &a2,const c_number4 &a3,const c_number4 &b1,const c_number4 &b2,const c_number4 &b3,const c_number4 &r,const c_number &rmod,const c_number totRad,c_number4 &F, c_number4 &T){
    if(rmod-totRad>patchyCut) return;
    for(int pi=0;pi<patchParticle[p][0];pi++){
                        // printf("Pi = %i\n",patchParticle[p][0]);
        c_number4 ppatch={
            a1.x*patch[patchParticle[p][pi+1]][2]+a2.x*patch[patchParticle[p][pi+1]][3]+a3.x*patch[patchParticle[p][pi+1]][4],
            a1.y*patch[patchParticle[p][pi+1]][2]+a2.y*patch[patchParticle[p][pi+1]][3]+a3.y*patch[patchParticle[p][pi+1]][4],
            a1.z*patch[patchParticle[p][pi+1]][2]+a2.z*patch[patchParticle[p][pi+1]][3]+a3.z*patch[patchParticle[p][pi+1]][4],
            0.0
        };
        for(int qi=0;qi<patchParticle[q][0];qi++){
            c_number4 qpatch={
                b1.x*patch[patchParticle[q][qi+1]][2]+b2.x*patch[patchParticle[q][qi+1]][3]+b3.x*patch[patchParticle[q][qi+1]][4],
                b1.y*patch[patchParticle[q][qi+1]][2]+b2.y*patch[patchParticle[q][qi+1]][3]+b3.y*patch[patchParticle[q][qi+1]][4],
                b1.z*patch[patchParticle[q][qi+1]][2]+b2.z*patch[patchParticle[q][qi+1]][3]+b3.z*patch[patchParticle[q][qi+1]][4],
                0.0
            };
            c_number4 patchDist={
                r.x-ppatch.x+qpatch.x,
                r.y-ppatch.y+qpatch.y,
                r.z-ppatch.z+qpatch.z,
                0.0
            };
            c_number dist = CUDA_DOT(patchDist,patchDist);
            if(dist>patchyCut2 || patch[patchParticle[p][pi+1]][0]+patch[patchParticle[q][qi+1]][0]!=0) continue; // for non reacheable and non complementary patches ignore
            c_number K = patch[patchParticle[p][pi+1]][1]+patch[patchParticle[q][qi+1]][1];
            c_number r2b2 = dist*alphaB2;
            c_number r8b10 = r2b2*r2b2*r2b2*r2b2*alphaB2;
            c_number energy = -1.0*exp(-1*r8b10*dist)*K;
            c_number4 force = patchDist*(10*r8b10*energy);
            c_number4 torque = _cross(ppatch,force);
            // printf("Energy %f\n",energy);
            F.w+=energy;
            F.x -= force.x;
            F.y -= force.y;
            F.z -= force.z;
            T.x -= torque.x;
            T.y -= torque.y;
            T.z -= torque.z;
        }
    }
}

__global__ void CUDAPSP2Particle(const c_number4 __restrict__ *poss, const GPU_quat __restrict__ *orientations, c_number4 __restrict__ *forces, c_number4 __restrict__ *torques, const int *matrix_neighs,const int *number_neighs, const CUDABox *box){
    const int ind =IND;
    if(ind >= MD_N[0]) return; // for i > N leave
    c_number4 F = forces[ind];
    c_number4 T = torques[ind];
	c_number4 ppos = poss[ind];
    c_number4 a1, a2, a3,b1,b2,b3;
    get_vectors_from_quat(orientations[ind], a1, a2, a3);

    for(int p=0;p<connection[ind][0];p++){
        int id = connection[ind][p+1];
        get_vectors_from_quat(orientations[id], b1, b2, b3);
        c_number4 r = box->minimum_image(ppos, poss[id]);
        TorqueSpring(ind,id,p,a1,a2,a3,b1,b2,b3,r,F,T);
    }

    int num_neighs = NUMBER_NEIGHBOURS(ind, number_neighs);
    for(int j = 0; j < num_neighs; j++) {
		int k_index = NEXT_NEIGHBOUR(ind, j, matrix_neighs);
        if(k_index == ind || strand[ind]==strand[k_index]) continue; // Skip self interaction and same strand interaction
        c_number4 r = box->minimum_image(ppos, poss[k_index]);
        get_vectors_from_quat(orientations[k_index], b1,b2,b3);
        c_number r2 = CUDA_DOT(r,r);
        c_number rmod = sqrtf(r2);
        c_number totRad = radius[ind]+radius[k_index];
        repulsiveLinear(r,r2,rmod,totRad,F);
        simplePatch(ind,k_index,a1,a2,a3,b1,b2,b3,r,rmod,totRad,F,T);
    }
    forces[ind] = F;
	torques[ind] = _vectors_transpose_c_number4_product(a1, a2, a3, T);

};

CUDAPSP2Interaction::CUDAPSP2Interaction():PSP2Interaction() {}
CUDAPSP2Interaction::~CUDAPSP2Interaction() {}
void CUDAPSP2Interaction::get_settings(input_file &inp) {
    PSP2Interaction::get_settings(inp);
}

void CUDAPSP2Interaction::cuda_init(int N){
    // Initialize the interaction
    // std::cout<<"Initializing the PSP2 interaction"<<std::endl;
    CUDABaseInteraction::cuda_init(N);
    PSP2Interaction::init();
    std::vector<BaseParticle *> particles(N);
    // std::cout<<"N = "<<N<<std::endl;
    int my_N_strands;
    PSP2Interaction::read_topology(&my_N_strands,particles);
    // std::cout<<"Starting to copy data to the device"<<std::endl;
    // Copy the data to the device
    // Integers
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(strand, &particleStrand, sizeof(int)*MAXparticles));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(connection, &connections, sizeof(int)*MAXparticles*(MAXneighbour+1)));

    // // Floats
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(radius, &particleRadius, sizeof(float)*MAXparticles));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(spring, &Springs, sizeof(float)*MAXSprings*5));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(patch, &Patches, sizeof(float)*MAXPatches*5));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(springParticle, &ParticleSprings, sizeof(int)*MAXparticles*MAXSpringPerParticle));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(invSpringParticle, &invParticleSprings, sizeof(int)*MAXparticles*MAXSpringPerParticle));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(patchParticle, &ParticlePatches, sizeof(int)*MAXparticles*(MAXPatchPerParticle+1)));

    //Singular Floats
    COPY_NUMBER_TO_FLOAT(sigma,patchySigma);
    COPY_NUMBER_TO_FLOAT(Rstar,patchyRstar);
    COPY_NUMBER_TO_FLOAT(patchyb,patchyB);
    COPY_NUMBER_TO_FLOAT(Rc,patchyRc);
    COPY_NUMBER_TO_FLOAT(patchyCut2,patchyRcut2);
    COPY_NUMBER_TO_FLOAT(patchyCut,patchyRcut);
    COPY_NUMBER_TO_FLOAT(alpha,patchyAlpha);
    COPY_NUMBER_TO_FLOAT(alphaB2,patchyAlphaB2);



    // std::cout<<"Data copied to the device"<<std::endl;
    // Delete particles to save space
    for(int i = 0; i < N; i++) {
        delete particles[i];
    }

    // Debugging
    // for(int i = 0; i < N; i++) {
    //     std::cout<<particleStrand[i]<<std::endl;
    // }
}

void CUDAPSP2Interaction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
    // Compute the forces per particles
    CUDAPSP2Particle <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>> 
    (d_poss, d_orientations, d_forces,  d_torques, lists->d_matrix_neighs, lists->d_number_neighs, d_box);
    // std::cout<< "Computing forces for PSP2"<<std::endl;
    CUT_CHECK_ERROR("Kernel failed, something quite exciting");
}