#ifndef PSP2Interaction_H_
#define PSP2Interaction_H_


#include "BaseInteraction.h"
#include "../Particles/PSP2Particle.h"
#include "../Utilities/parse_input/parse_input.h"
#include <fstream>
#include <sstream>
#include <cmath>

#define MAXparticles 3000 // maximum number of particles
#define MAXPatches 50
#define MAXSprings 100
#define MAXPatchPerParticle 1
#define MAXSpringPerParticle 12
#define MAXneighbour 12 // number of bonded neighbout one particle is connected t0

class PSP2Interaction: public BaseInteraction {
protected:
public:
	number rnorm, rmod;
    int particleNum,strands,maxPatches,maxSprings; // header
	number patchySigma=1.0f,patchyRstar=0.9053f,patchyRc=0.99998,patchyB=667.505671539,patchyRcut=1.2,patchyAlpha=0.12,springMultiplier=2;
	number patchyRcut2 = SQR(patchyRcut), patchyAlphaB2 = 1/SQR(patchyAlpha);
	float particleRadius[MAXparticles];
	int particleStrand[MAXparticles];
	float Patches[MAXPatches][5]; // color,strength,x,y,z
	float Springs[MAXSprings][5]; // k, ro , x,y,z
	int ParticlePatches[MAXparticles][MAXPatchPerParticle+1];
	int connections[MAXparticles][MAXneighbour+1];
	int ParticleSprings[MAXparticles][MAXSpringPerParticle];
	int invParticleSprings[MAXparticles][MAXSpringPerParticle];
	float SpringR0[MAXparticles][MAXSpringPerParticle];
	std::string temp;




	PSP2Interaction();
	virtual ~PSP2Interaction();

	number ccg_interaction_bonded(BaseParticle *p, BaseParticle *q, bool compute_r = true, bool update_forces = false){
		OX_DEBUG("Bonded interaction is called");
		number energy =0.f;
		
		energy+=25;
		return energy;
	};
	// Necessary interaction
    virtual void get_settings(input_file &inp);
	virtual void init();
	virtual void allocate_particles(std::vector<BaseParticle *> &particles); //Add particle to the system
	virtual void read_topology(int *N_strands, std::vector<BaseParticle *> &particles); // Read the top file
	virtual void check_input_sanity(std::vector<BaseParticle *> &particles); // Check all the input file are correct.

	//Interaction that are updated repeatedly
	virtual number pair_interaction(BaseParticle *p, BaseParticle *q, bool compute_r = true, bool update_forces = false); //Check bonded or non-bonded
	virtual number pair_interaction_bonded(BaseParticle *p, BaseParticle *q, bool compute_r = true, bool update_forces = false); // Bonded particle interaction
	virtual number pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q, bool compute_r = true, bool update_forces = false); //Non-bonded particle interaction

	// Bonded Interactions

	virtual number torqueSpring(BaseParticle *p, BaseParticle *q, bool compute_r = true, bool update_forces = false);
	// virtual number Spring(BaseInteraction *p, BaseInteraction *q, bool compute_r, bool update_forces);

	// Non-bonded Interactions
	virtual number exeVol(BaseParticle *p, BaseParticle *q, bool compute_r = true, bool update_forces = false);
	virtual number repulsiveLinear(number prefactor, const LR_vector &r, LR_vector &force, number sigma, number rstar, number b, number rc, bool update_forces); //Excluded volume with pre-factor
	virtual number simplePatch(BaseParticle *p, BaseParticle *q, bool compute_r = true, bool update_forces = false);



	//Common Functions
	bool bonded(BaseParticle *p, BaseParticle *q);
	int returnKro(int p, int q, number *k, number *r0, LR_vector *rp, LR_vector *rq);
	int returnKro(int p, int q, number *k, number *r0, LR_vector *rp, LR_vector *rq, int next);
	LR_vector returnPatch(int p,int i); 
	// Extra Topology Functions

	void populateInvSprings();
	// void populateSpringR0();
		
};

//Debug functions
template <typename T> void print2DArray(T* arr, int rows, int cols);







#endif /* PSP2Interaction_H_ */