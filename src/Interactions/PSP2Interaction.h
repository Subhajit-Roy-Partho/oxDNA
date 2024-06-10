#ifndef PSP2Interaction_H_
#define PSP2Interaction_H_


#include "BaseInteraction.h"
#include "../Particles/PSP2Particle.h"
#include "../Utilities/parse_input/parse_input.h"
#include <fstream>
#include <sstream>
#include <cmath>

#define MAXparticles 5000 // maximum number of particles
#define MAXPatches 1
#define MAXSprings 2
#define MAXPatchPerParticle 1
#define MAXSpringPerParticle 12
#define MAXneighbour 5 // number of bonded neighbout one particle is connected t0

class PSP2Interaction: public BaseInteraction {
protected:
public:
    int particleNum,strands,maxPatches,maxSprings; // header

	number particleRadius[MAXparticles];
	number particleStrand[MAXparticles];
	number Patches[MAXPatches][5];
	number Springs[MAXSprings][5];
	number ParticlePatches[MAXparticles][MAXPatchPerParticle+1];
	number ParticleSprings[MAXparticles][MAXSpringPerParticle+1];
	int connections[MAXparticles][MAXneighbour+1];



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
};







#endif /* PSP2Interaction_H_ */