#include "CUDAPHBInteraction.h"

__constant__ float rcut2;  // cut-off distance squared
__constant__ int exclusionType; // 0 for Linear, 1 for Cubic, 2 for Hard
__constant__ float sigma; // 
__constant__ float patchyb; // Controls the stiffness of exe volume and in case of hard the power over (sigma/r).
__constant__ float Rstar;
__constant__ float Rc;
__constant__ float patchyRcutSqr;
__constant__ float powAlpha;
__constant__ float patchyEps;
__constant__ float hardVolCutoff;
__constant__ int MD_N[1];
__constant__ int n_forces;
__constant__ float B2;
__constant__ float invPowAlpha;
__constant__ float patchLockCut;

__constant__ float patches[GPUmaxiP][5];// color,strength,x,y,z
__constant__ int numPatches[GPUmaxiC][MaxPatches];// num, patch1, patch2, patch3, patch4, patch5, patch6

__constant__ float ka;
__constant__ float kb;
__constant__ float kt;

__device__ int connection[MAXparticles][MAXneighbour+1];//first intger will state number of connections
__device__ float ro[MAXparticles][MAXneighbour];//rarius of the spring
__device__ float k[MAXparticles][MAXneighbour];//spring constant
__device__ int strand[MAXparticles];
__device__ int iC[MAXparticles];
__device__ float radius[MAXparticles];
__device__ float mass[MAXparticles];
__device__ int PatchLock[MAXparticles][MaxPatches][2]; //particle, patchid

// Particle properties made constant
__constant__ float tu=0.952319757;
__constant__ float tk=1.14813301;
__constant__ float kb1=1;
__constant__ float kb2=0.80;


__device__ void rotateVectorAroundVector(c_number4 &v, c_number4 &axis, c_number angle)
{
    // c_number4 tmp = axis * CUDA_DOT(v, axis);
    // v -= tmp;
    // c_number4 w = _cross(axis, v);
    // v = tmp + v * cosf(angle) + w * sinf(angle);

    c_number costh = cos(angle);
    c_number sinth = sin(angle);
    c_number scalar = CUDA_DOT(v, axis);
    c_number4 cross = _cross(axis, v);
    v.x = axis.x*scalar*(1.-costh)+v.x*costh+cross.x*sinth;
    v.y = axis.y*scalar*(1.-costh)+v.y*costh+cross.y*sinth;
    v.z = axis.z*scalar*(1.-costh)+v.z*costh+cross.z*sinth;
}

////////////////////bonded Interactions //////////////////////

__device__ void CUDAspring(c_number4 &r, c_number r0, c_number k, c_number4 &F, c_number &rmod)
{
    c_number dist = rmod - r0;
    F.w += 0.5f * k * dist * dist;
    c_number magForce = (k * dist) / rmod;
    F.x += r.x * magForce;
    F.y += r.y * magForce;
    F.z += r.z * magForce;
};

///////////////////nonbonded Interactions ////////////////////

///////////////////Voulume Exclusion /////////////////////////

// __device__ void CUDArepulsive_lj2(c_number prefactor, c_number4 &r, c_number4 &F, c_number sigma, c_number rstar, c_number b, c_number rc) {
// 	c_number rnorm = CUDA_DOT(r, r);
// 	if(rnorm < SQR(rc)) {
// 		if(rnorm > SQR(rstar)) {
// 			c_number rmod = sqrtf(rnorm);
// 			c_number rrc = rmod - rc;
// 			c_number part = prefactor * b * rrc;
// 			F.x += r.x * (2.f * part / rmod);
// 			F.y += r.y * (2.f * part / rmod);
// 			F.z += r.z * (2.f * part / rmod);
// 			F.w += part * rrc;
// 		}
// 		else {
// 			c_number tmp = SQR(sigma) / rnorm;
// 			c_number lj_part = tmp * tmp * tmp;
// 			// the additive term was added by me to mimick Davide's implementation
// 			F.x += r.x * (24.f * prefactor * (lj_part - 2.f * SQR(lj_part)) / rnorm);
// 			F.y += r.y * (24.f * prefactor * (lj_part - 2.f * SQR(lj_part)) / rnorm);
// 			F.z += r.z * (24.f * prefactor * (lj_part - 2.f * SQR(lj_part)) / rnorm);
// 			F.w += 4.f * prefactor * (SQR(lj_part) - lj_part) + prefactor;
// 		}
// 	}
// }


// CPU hits stuff twice for each pair of particles
__device__ void CUDAexeVolLin(c_number prefactor, c_number4 &r, c_number4 &F, c_number sigma, c_number rstar, c_number b, c_number rc,c_number r2)
{
    // auto r2 = CUDA_DOT(r, r);
    if (r2 < SQR(rc))
    {
        if (r2 > SQR(rstar))
        {
            c_number rmod = sqrtf(r2);
            c_number rrc = rmod - rc;
            c_number fmod = 4.f * prefactor * b * rrc / rmod;
            F.x += r.x * fmod;
            F.y += r.y * fmod;
            F.z += r.z * fmod;
            F.w += prefactor * b * SQR(rrc)*2;
        }else{
            c_number lj_part = CUB(SQR(sigma) / r2);
            c_number fmod = 48.f * prefactor * (lj_part - 2.f * SQR(lj_part)) / r2;
            F.x += r.x * fmod;
            F.y += r.y * fmod;
            F.z += r.z * fmod;
            F.w += 8.f * prefactor * (SQR(lj_part) - lj_part);
        }
    }
}

///////////////////Helix Interaction /////////////////////////

// __device__ void CUDAbondedTwist(c_number4 &fp, c_number4 &fq, c_number4 &vp, c_number4 &vq, c_number4 &up, c_number4 &uq)
// {
//     c_number cos_alpha_plus_gamma = CUDA_DOT(fp, fq) + CUDA_DOT(vp, vq) / 1 + CUDA_DOT(up, uq);
// };

// __device__ void CUDAbondedAlignment(c_number4 &tp, c_number4 &up,c_number4 &F, c_number4 &T,c_number &rmod,bool straight){
//     int factor=1;
//     if(!straight){ tp = -tp; factor=-1;}; // factor make sure that the force is negative if the alignment is not opposite
//     c_number4 force = factor*ka*(up-tp*CUDA_DOT(up,tp)/SQR(rmod))/rmod;
//     F.x += force.x;
//     F.y += force.y;
//     F.z += force.z;
//     F.w += ka*(1-CUDA_DOT(up,tp)/rmod);
//     c_number4 torque = -(ka*_cross(tp,up))/rmod;
//     T.x += torque.x;
//     T.y += torque.y;
//     T.z += torque.z;
// }

__device__ void CUDAbondedDoubleBending(c_number4 &up, c_number4 &uq,c_number4 &F, c_number4 &T){
    c_number4 torque;

    c_number cosine = CUDA_DOT(up,uq);
    c_number gu = cosf(tu);
    c_number angle = LRACOS(cosine);

    c_number A = (kb2*sinf(tk)-kb1*sinf(tu))/(tk-tu);
    c_number B = kb1 * sinf(tu);
    c_number C = kb1 * (1.f - gu) - (A * SQR(tu) / 2.f + tu * (B - tu * A));
    c_number D = cosf(tk) + (A * SQR(tk) * 0.5f + tk * (B - tu * A) + C) / kb2;

	c_number g1 = (angle - tu) * A + B;
	c_number g_ = A * SQR(angle) / 2.f + angle * (B - tu * A);
	c_number g = g_ + C;

    // unkinked bending regime
    if(angle < tu){
        torque = kb * kb1 * _cross(up, uq);
		F.w += kb * (1.f - cosine) * kb1;
    }
    // intermediate regime
    else if(angle < tk) {
		c_number4 sin_vector = _cross(up, uq);
		c_number sin_module = _module(sin_vector);
		sin_vector.x /= sin_module;
		sin_vector.y /= sin_module;
		sin_vector.z /= sin_module;

		torque = kb * g1 * sin_vector;
		F.w += kb * g;
	}
    // kinked bending regime - same as unkinked, but with an additive term to the energy and with a different bending.
    else {
		torque= kb * kb2 * _cross(up, uq);
		F.w += kb * (D - cosine) * kb2;
	}
    T.x+=torque.x;
    T.y+=torque.y;
    T.z+=torque.z;
}

// __device__ void bondedInteraction(c_number4 &poss ,c_number4 &a1, c_number4 &a2, c_number4 &a3,c_number4 &qoss, c_number4 &b1, c_number4 &b2, c_number4 &b3,c_number4 &F, c_number4 &T, CUDABox *box, int pid,int qid,c_number &ro,c_number &k){
//     c_number4 r = box->minimum_image(poss, qoss);
//     c_number rmod = sqrt(CUDA_DOT(r, r));
    
//     CUDAspring(r,ro,k,F,rmod);
//     // printf("Mag of force = %d\n",F.w);
//     CUDAbondedDoubleBending(a1,b1,F,T);
//     // if(qid-pid==1){
//     //     CUDAbondedAlignment(r,a1,F,T,rmod,false);
//     // }else if(pid-qid==1){
//     //     CUDAbondedAlignment(r,b1,F,T,rmod,true);
//     // }
// }

// __device__ void nonbondedInteraction(c_number4 &ppos ,c_number4 &a1, c_number4 &a2, c_number4 &a3,c_number4 &qpos, c_number4 &b1, c_number4 &b2, c_number4 &b3,c_number4 &F, c_number4 &T, CUDABox *box,c_number totRad,int pi,int pj){
//     c_number4 r = box->minimum_image(ppos, qpos);
//     c_number sqr_r = CUDA_DOT(r, r);
//     if(sqr_r>rcut2) return;
    
//     CUDAexeVolLin(1.f,r,F,sigma*totRad,Rstar*totRad,patchyb/SQR(totRad),Rc*totRad,sqr_r);
//     CUDApatchySimple(r,a1,a2,a3,b1,b2,b3,F,T,pi,pj);
// }

__global__ void CUDAparticle(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, int *matrix_neighs, int *number_neighs, CUDABox *box){
    // printf("IND = %d\n",IND);
    // printf("Initial force value F.w = %f\n",forces[IND].w);
    // printf("Lock cut = %f\n",patchLockCut);
    // printf("Patch Lock = %d\n",PatchLock[IND][0][0]);
    if(IND >= MD_N[0]) return; // for i > N leave
    c_number4 F = forces[IND];
    c_number4 T = torques[IND];
	c_number4 ppos = poss[IND];
    // printf("Position check = (%f,%f,%f,%f)\n",ppos.x,ppos.y,ppos.z,ppos.w);
    // printf("Sigma = %f\n",rcut2);
	c_number4 a1, a2, a3;
	get_vectors_from_quat(orientations[IND], a1, a2, a3);

    //Bonded Interactions
    for(int p =0;p<connection[IND][0];p++){
        int id = connection[IND][p+1];
        c_number4 b1, b2, b3;
        get_vectors_from_quat(orientations[id], b1, b2, b3);
        // printf("p= %i q= %i  r0= %f  k= %f\n",IND,id,ro[IND][p],k[IND][p]);
        c_number4 r = box->minimum_image(ppos, poss[id]);
        c_number sqr_r = CUDA_DOT(r, r);
        c_number rmod = sqrtf(sqr_r);
        CUDAspring(r,ro[IND][p],k[IND][p],F,rmod);
        

        CUDAbondedDoubleBending(a1,b1,F,T);
        // bool straight;
        c_number4 up,tp;
        if(id-IND==1){
            up = a1;
            tp = r;

            c_number4 force = ka*(up-tp*CUDA_DOT(up,tp)/SQR(rmod))/rmod;
            F.x -= force.x;
            F.y -= force.y;
            F.z -= force.z;
            F.w += ka*(1.f-CUDA_DOT(up,tp)/rmod);
            c_number4 torque = -(ka*_cross(tp,up))/rmod;
            T.x += torque.x;
            T.y += torque.y;
            T.z += torque.z;
        }

        if(IND-id==1){
            up = b1;
            tp = -r;

            c_number4 force = ka*(up-tp*CUDA_DOT(up,tp)/SQR(rmod))/rmod;
            F.x += force.x;
            F.y += force.y;
            F.z += force.z;
            F.w += ka*(1.f-CUDA_DOT(up,tp)/rmod);
                // c_number4 torque = -(ka*_cross(tp,up))/rmod;
                // T.x += torque.x;
                // T.y += torque.y;
                // T.z += torque.z;
        }
    }



    // for(int j=0;j<numPatches[IND][0];j++){
    //     printf("i = %i  j = %i  patches = %i \n",IND,j,numPatches[IND][j+1]);
    // }

    // printf("i=%i  iC=%i\n",IND,iC[IND]);

    // Non bonded interactions
    int num_neighs = NUMBER_NEIGHBOURS(IND, number_neighs);
    for(int j = 0; j < num_neighs; j++) {
		int k_index = NEXT_NEIGHBOUR(IND, j, matrix_neighs);

        if(k_index != IND && strand[IND]!=strand[k_index]) {
            // printf("For index = %i, kIndex = %i\n",IND,k_index);
			c_number4 qpos = poss[k_index];
            c_number4 b1, b2, b3;
            get_vectors_from_quat(orientations[k_index], b1, b2, b3);
            c_number totalRad = radius[IND] + radius[k_index];
            c_number4 r = box->minimum_image(ppos, qpos);
            c_number sqr_r = CUDA_DOT(r, r);
            // r.w = sqrt(sqr_r);
            // printf("p= %i  q= %i  radius= (%f,%f,%f,%f)\n",IND,k_index,r.x,r.y,r.z,r.w);
            c_number rSigma = totalRad*sigma;
            c_number rRstar = totalRad*Rstar;
            c_number rb = patchyb/SQR(totalRad);
            c_number rRc = totalRad*Rc;
            CUDAexeVolLin(1.f,r,F,rSigma,rRstar,rb,rRc,sqr_r);

            // Patchy Interaction //
            int pParticleColor = iC[IND];
            int qParticleColor = iC[k_index];
            if(pParticleColor !=100 && qParticleColor !=100 && (sqr_r-totalRad*totalRad)<patchyRcutSqr){ //This speed up stuff
                // printf("i=%i  j=%i\n",IND,k_index);
                for(int pi=0;pi<numPatches[pParticleColor][0];pi++){
                    c_number4 ppatch ={
                        a1.x*patches[numPatches[pParticleColor][pi+1]][2]+a2.x*patches[numPatches[pParticleColor][pi+1]][3]+a3.x*patches[numPatches[pParticleColor][pi+1]][4],
                        a1.y*patches[numPatches[pParticleColor][pi+1]][2]+a2.y*patches[numPatches[pParticleColor][pi+1]][3]+a3.y*patches[numPatches[pParticleColor][pi+1]][4],
                        a1.z*patches[numPatches[pParticleColor][pi+1]][2]+a2.z*patches[numPatches[pParticleColor][pi+1]][3]+a3.z*patches[numPatches[pParticleColor][pi+1]][4],
                        0.f
                    };
                    // printf("i=%i  j=%i  ppatch= (%f,%f,%f,%f)",IND,k_index,ppatch.x,ppatch.y,ppatch.z);
                    for(int qj=0;qj<numPatches[qParticleColor][0];qj++){
                    //     // printf("i=%i  j=%i\n",IND,k_index);
                        c_number4 qpatch={
                            b1.x*patches[numPatches[qParticleColor][qj+1]][2]+b2.x*patches[numPatches[qParticleColor][qj+1]][3]+b3.x*patches[numPatches[qParticleColor][qj+1]][4],
                            b1.y*patches[numPatches[qParticleColor][qj+1]][2]+b2.y*patches[numPatches[qParticleColor][qj+1]][3]+b3.y*patches[numPatches[qParticleColor][qj+1]][4],
                            b1.z*patches[numPatches[qParticleColor][qj+1]][2]+b2.z*patches[numPatches[qParticleColor][qj+1]][3]+b3.z*patches[numPatches[qParticleColor][qj+1]][4],
                            0.f
                        };
                    //     // printf("i=%i  j=%i  qpatch= (%f,%f,%f,%f)\nS",IND,k_index,qpatch.x,qpatch.y,qpatch.z);
                        c_number4 patchDist ={
                            r.x+qpatch.x-ppatch.x,
                            r.y+qpatch.y-ppatch.y,
                            r.z+qpatch.z-ppatch.z,
                            0.f
                        };
                        c_number dist = CUDA_DOT(patchDist,patchDist);
                        if((patches[numPatches[pParticleColor][pi+1]][0]+patches[numPatches[qParticleColor][qj+1]][0]==0 && dist<patchyRcutSqr)){
                            if((PatchLock[IND][pi][0]==-1||(PatchLock[IND][pi][0]==k_index && PatchLock[IND][pi][1]==qj)) && (PatchLock[k_index][qj][0]==-1||(PatchLock[k_index][qj][0]==IND && PatchLock[k_index][qj][1]==pi))){
                                c_number strength = patches[numPatches[pParticleColor][pi+1]][1]+patches[numPatches[qParticleColor][qj+1]][1];
                                // c_number r2b2 = dist*B2;
                                // c_number r8b10 = r2b2*r2b2*r2b2*r2b2*B2;
                                c_number r8b10 = dist*dist*dist*dist*invPowAlpha;
                                c_number exp_part = -1.f*expf(-1.f*r8b10*dist)*strength;
                                c_number4 tmp_force = patchDist*(10.f*exp_part*r8b10); //modified

                                c_number4 torque = _cross(ppatch,tmp_force);
                                F.w+= exp_part;
                                F.x -= tmp_force.x;
                                F.y -= tmp_force.y;
                                F.z -= tmp_force.z;
                                T.x -= torque.x;
                                T.y -= torque.y;
                                T.z -= torque.z;
                                if(exp_part<patchLockCut){
                                    // if(PatchLock[IND][pi][0]==-1 || PatchLock[k_index][qj][0]==-1){ // this is slow because of multiple calls
                                        PatchLock[IND][pi][0] = k_index;
                                        PatchLock[IND][pi][1] = qj;
                                        PatchLock[k_index][qj][0] = IND;
                                        PatchLock[k_index][qj][1] = pi;
                                    // }
                                }else{
                                    PatchLock[IND][pi][0] = -1;
                                    PatchLock[IND][pi][1] = -1;
                                    PatchLock[k_index][qj][0] = -1;
                                    PatchLock[k_index][qj][1] = -1;
                                }
                            } else if((PatchLock[IND][pi][0]==k_index && PatchLock[IND][pi][1]==qj)&&(PatchLock[k_index][qj][0]!=IND || PatchLock[k_index][qj][1]!=pi)){
                                PatchLock[IND][pi][0] = -1;
                                PatchLock[IND][pi][1] = -1;
                                PatchLock[k_index][qj][0] = -1;
                                PatchLock[k_index][qj][1] = -1;
                                printf("Assymetrically locked, unlocking\n");
                            }
                        }
                    }
                }
            }
        }
    }

    // printf("Force = (%f,%f,%f,%f)\n",F.x,F.y,F.z,F.w);
    forces[IND] = F;
	torques[IND] = _vectors_transpose_c_number4_product(a1, a2, a3, T);
}

CUDAPHBInteraction::CUDAPHBInteraction():PHBInteraction(){}
CUDAPHBInteraction::~CUDAPHBInteraction(){}
void CUDAPHBInteraction::get_settings(input_file &inp){
    PHBInteraction::get_settings(inp);
}
void CUDAPHBInteraction::cuda_init(int N){
    CUDABaseInteraction::cuda_init(N);
    PHBInteraction::init();
    // std::cout<<"N = "<<N<<std::endl;
    std::vector<BaseParticle *> particles(N);
    PHBInteraction::allocate_particles(particles);
	int my_N_strands;
	PHBInteraction::read_topology(&my_N_strands, particles);
    // std::cout<<this->_rcut<<std::endl;
    // std::cout<<this->_sqr_rcut<<std::endl;

    for(int i = 0; i < N; i++) {
		delete particles[i];
	}

    number r8b10 = powf(patchyRcut, (number) 8.f) / patchyPowAlpha;
    number GPUhardVolCutoff = -1.001f * exp(-(number) 0.5f * r8b10 * patchyRcut2);

    int tempPatchLock[MAXparticles][MaxPatches][2];
    for(int i=0;i<MAXparticles;i++){
        for(int j=0;j<MaxPatches;j++){
            for(int k=0;k<2;k++)
            tempPatchLock[i][j][k] = -1;
        }
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(PatchLock, &tempPatchLock, sizeof(int) * MAXparticles * MaxPatches*2));

    float invPow2 = 1.f / SQR(patchyAlpha);
    // float invAlpha = patchyAlpha*patchyAlpha*patchyAlpha*patchyAlpha*patchyAlpha;
    COPY_NUMBER_TO_FLOAT(invPowAlpha, invPatchyPowAlpha);
    COPY_NUMBER_TO_FLOAT(B2, invPow2);
    // std::cout<<"Patchy Alpha"<<patchyAlpha<<std::endl;
    // std::cout<<"Inverse patchy Alpha square"<<invPow2<<std::endl;
    // std::cout<<"Inverse patchy Pow alpha"<<invPatchyPowAlpha<<std::endl;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
    COPY_NUMBER_TO_FLOAT(rcut2, _sqr_rcut);
    COPY_NUMBER_TO_FLOAT(sigma, patchySigma);
    COPY_NUMBER_TO_FLOAT(patchyb, patchyB);
    COPY_NUMBER_TO_FLOAT(Rstar, patchyRstar);
    COPY_NUMBER_TO_FLOAT(Rc, patchyRc);
    COPY_NUMBER_TO_FLOAT(patchyRcutSqr, patchyRcut2);
    COPY_NUMBER_TO_FLOAT(powAlpha, patchyPowAlpha);
    // std::cout<<"Patchy Power Alpha = "<<patchyPowAlpha<<std::endl;

    COPY_NUMBER_TO_FLOAT(ka, _ka);
    COPY_NUMBER_TO_FLOAT(kb, _kb);
    COPY_NUMBER_TO_FLOAT(kt, _kt);
    COPY_NUMBER_TO_FLOAT(patchLockCut,patchyLockCutOff);

    // COPY_ARRAY_TO_CONSTANT(patches,&GPUnumPatches,GPUmaxiP*5);
    // COPY_ARRAY_TO_CONSTANT(connection, GPUconnections, MAXparticles * (MAXneighbour + 1));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(patchyb, &patchyB, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(Rstar, &patchyRstar, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(Rc, &patchyRc, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(patchyRcutSqr, &patchyCutOff2, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(powAlpha, &patchyPowAlpha, sizeof(float)));


    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(patchyEps, &patchyEpsilon, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(hardVolCutoff, &GPUhardVolCutoff, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(n_forces, &_n_forces, sizeof(int)));


    CUDA_SAFE_CALL(cudaMemcpyToSymbol(patches, &GPUpatches, sizeof(float) * GPUmaxiP * 5));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(numPatches, &GPUnumPatches, sizeof(int) * GPUmaxiC * MaxPatches));

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(connection, &GPUconnections, sizeof(int) * MAXparticles * (MAXneighbour + 1)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, &GPUro, sizeof(float) * MAXparticles * MAXneighbour));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k, &GPUk, sizeof(float) * MAXparticles * MAXneighbour));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(strand, &GPUstrand, sizeof(int) * MAXparticles));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(iC, &GPUiC, sizeof(int) * MAXparticles));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(radius, &GPUradius, sizeof(float) * MAXparticles));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(mass, &GPUmass, sizeof(float) * MAXparticles));

    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(ka, &_ka, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(kb, &_kb, sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemcpyToSymbol(kt, &_kt, sizeof(float)));

    // std::cout<<"CUDA Connection Info"<<std::endl;
    // for(int i=0;i<7;i++){
    //     for(int j=0;j<GPUconnections[i][0];j++){
    //         std::cout<<GPUconnections[i][j+1]<<"\t";
    //         std::cout<<GPUro[i][j]<<"\t";
    //         std::cout<<GPUk[i][j]<<"\t";
    //     }
    //     std::cout<<"\n";
    // }

    // std::cout<<"CUDA patch Info"<<std::endl;
    // for(int i=0;i<GPUmaxiP;i++){
    //     for(int j=0;j<5;j++){
    //         std::cout<<GPUpatches[i][j]<<"\t";
    //     }
    //     std::cout<<"\n";
    // }

    // std::cout<<"CUDA particle color Information"<<std::endl;
    // for(int i=0;i<GPUmaxiC;i++){
    //     for(int j=0;j<GPUnumPatches[i][0];j++){
    //         std::cout<<GPUnumPatches[i][j+1]<<"\t";
    //     }
    //     std::cout<<"\n";
    // }
    // std::cout<<patchyPowAlpha<<std::endl;
}

void CUDAPHBInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box)
{
    // CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);

    // if(_update_st) CUDA_SAFE_CALL(cudaMemset(_d_st, 0, _N * sizeof(CUDAStressTensor)));
    // if(_n_forces == 1){
    //     CUDAnonbondedParticles
    //     <<<(lists->N_edges - 1)/(_launch_cfg.threads_per_block) + 1, _launch_cfg.threads_per_block>>>
    //     (d_poss, d_orientations, d_forces, d_torques, lists->d_edge_list, lists->N_edges, _update_st, _d_st, d_box);
    // }else{
    //     CUDAnonbondedParticles<<<(lists->N_edges - 1)/(_launch_cfg.threads_per_block) + 1, _launch_cfg.threads_per_block>>>
    //     (d_poss, d_orientations, _d_edge_forces, _d_edge_torques, lists->d_edge_list, lists->N_edges, _update_st, _d_st, d_box);

    //     _sum_edge_forces_torques(d_forces, d_torques);
    // }
    // CUDAbondedParticles<<<(lists->N_edges - 1)/(_launch_cfg.threads_per_block) + 1, _launch_cfg.threads_per_block>>>
    // (d_poss, d_orientations, d_forces, d_torques, _update_st, _d_st);
    // Partiles disctributes in the grid

    // std::cout<<d_poss[0].x<<d_poss[0].y<<d_poss[0].z<<std::endl;
    CUDAparticle <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>> 
    (d_poss, d_orientations, d_forces,  d_torques, lists->d_matrix_neighs, lists->d_number_neighs, d_box);

    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 0);
    // printf("Device %d has compute capability %d.%d.\n",
    // 0, deviceProp.major, deviceProp.minor);

    CUT_CHECK_ERROR("Kernel failed, something quite exciting");
}