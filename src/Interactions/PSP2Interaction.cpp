#include "PSP2Interaction.h"

PSP2Interaction::PSP2Interaction():BaseInteraction() {
}

PSP2Interaction::~PSP2Interaction() {

}

void PSP2Interaction::get_settings(input_file &inp) {
	BaseInteraction::get_settings(inp);
}

void PSP2Interaction::init() {
}

void PSP2Interaction::allocate_particles(std::vector<BaseParticle *> &particles) {
    particles.resize(particleNum);
    for(int i=0;i<particleNum;i++) {
        particles[i] = new PSP2Particle();
        particles[i]->index=i;
        particles[i]->strand_id=0;
        particles[i]->type=-5;
    }
}

void PSP2Interaction::read_topology(int *N_strands, std::vector<BaseParticle *> &particles) {
    std::string line,temp;
    std::ifstream topology;
    topology.open(this->_topology_filename,std::ios::in);
    if(!topology.good()) throw oxDNAException("Topology file not found");

    // Reading header
    std::getline(topology,line);
    std::istringstream head(line);
    head>> particleNum >> strands >> maxPatches >> maxSprings;
    *N_strands = strands;
    allocate_particles(particles);
    int i=0,m=0,n=0;
    while(std::getline(topology,line)) {
        if(line.size()==0 || line[0]=='#') continue;
        if(line[0]=='i'){
            std::istringstream body(line);
            body>>temp;
            if(temp=="iP"){
                body>>temp;
                body>>Patches[m][0]>>Patches[m][1]>>Patches[m][2]>>Patches[m][3]>>Patches[m][4];
                m++;
            }else if(temp=="iS"){
                body>>temp;
                body>>Springs[n][0]>>Springs[n][1]>>Springs[n][2]>>Springs[n][3]>>Springs[n][4];
                n++;
            }
            continue;
        }
        if(i>=particleNum) throw oxDNAException("Topology file has more particles than specified in the header");
        std::istringstream body(line);
        // int j=0;
        body>>particles[i]->type>>particleStrand[i]>>particleRadius[i];
        auto *pp = dynamic_cast<PSP2Particle *>(particles[i]);
        body>>ParticlePatches[i][0];
        for(int p=0;p<ParticlePatches[i][0];p++){
            body>>ParticlePatches[i][p+1];
        }
        int t=0;
        while(body.tellg()!=-1){
            body>>connections[i][t+1];
            body>>ParticleSprings[i][t];
            pp->add_neighbour(particles[connections[i][t+1]]); //needed for CPU bonded interaction
            t++;
        }
        connections[i][0]=t;
        i++;
    }

    // for(i=0;i<particleNum;i++){
    //     for(int j=0;j<connections[i][0];j++){
    //         std::cout<<connections[i][j+1]<<" "<<ParticleSprings[i][j]<<"\t";
    //     }
    //     std::cout<<"\n";
    // };

    // for(i=0;i<particleNum;i++){
    //     for(int j=0;j<ParticlePatches[i][0];j++){
    //         std::cout<<ParticlePatches[i][j+1]<<"\t";
    //     }
    //     std::cout<<"\n";
    // }

    // for(i=0;i<particleNum;i++){
    //     std::cout<<particleStrand[i]<<"\n";
    // }

    // for(i=0;i<MAXPatches;i++){
    //     for(int j=0;j<5;j++){
    //         std::cout<<Patches[i][j]<<"\t";
    //     }
    //     std::cout<<"\n";
    // }

    // for(i=0;i<MAXSprings;i++){
    //     for(int j=0;j<5;j++){
    //         std::cout<<Springs[i][j]<<"\t";
    //     }
    //     std::cout<<"\n";
    // }

    // std::cout<<bonded(particles[0],particles[1])<<"\n";
    // std::cout<<bonded(particles[0],particles[2])<<"\n";
    // std::cout<<bonded(particles[2],particles[3])<<"\n";
}

void PSP2Interaction::check_input_sanity(std::vector<BaseParticle *> &particles) {

}

number PSP2Interaction::pair_interaction(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    if(bonded(p,q)){
        return pair_interaction_bonded(p,q,compute_r,update_forces);
    }else{
        return pair_interaction_nonbonded(p,q,compute_r,update_forces);
    }
}

number PSP2Interaction::pair_interaction_bonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    if(compute_r) _computed_r = _box->min_image(p->pos, q->pos);
    number energy = torqueSpring(p,q,compute_r,update_forces);
    return energy;
}

number PSP2Interaction::pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    if(compute_r) _computed_r = _box->min_image(p->pos, q->pos);
    number energy = exeVol(p,q,compute_r,update_forces);
    energy+=simplePatch(p,q,compute_r,update_forces);
    return energy;
}

//Execlude Voulme

number PSP2Interaction::repulsiveLinear(number prefactor, const LR_vector &r, LR_vector &force, number sigma, number rstar, number b, number rc, bool update_forces) {
	// this is a bit faster than calling r.norm()
	rnorm = SQR(r.x) + SQR(r.y) + SQR(r.z);
	number energy = (number) 0;
	if(rnorm < SQR(rc)) {
		if(rnorm > SQR(rstar)) {
			rmod = sqrt(rnorm);
			number rrc = rmod - rc;
			energy = prefactor * b * SQR(rrc);
			if(update_forces)
				force = -r * (2 * prefactor * b * rrc / rmod);
		}
		else {
			number tmp = SQR(sigma) / rnorm;
			number lj_part = tmp * tmp * tmp;
			energy = 4 * prefactor * (SQR(lj_part) - lj_part);
			if(update_forces)
				force = -r * (24 * prefactor * (lj_part - 2 * SQR(lj_part)) / rnorm);
		}
	}

	if(update_forces && energy == (number) 0)
		force.x = force.y = force.z = (number) 0;

	return energy;
};

number PSP2Interaction::exeVol(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    number totRad = particleRadius[p->index] + particleRadius[q->index];
    number sigma = totRad*patchySigma;
    number rstar = totRad*patchyRstar;
    number b = patchyB/SQR(totRad);
    number rc = patchyRc*totRad;
    LR_vector force={0,0,0};
    number energy=repulsiveLinear(1,_computed_r,force,sigma,rstar,b,rc,update_forces);
    if(update_forces){
        p->force-=force;
        q->force+=force;
    }
    return energy;
}

//Simple Patch

number PSP2Interaction::simplePatch(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    return 0;
}

//Spring Function

number PSP2Interaction::torqueSpring(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    // for(int i=0;i<)
    LR_vector intCenter = p->orientation*p->pos;
    return 0;
}

//Common Functions

bool PSP2Interaction::bonded(BaseParticle *p, BaseParticle *q) {
    for(int i=0;i<connections[p->index][0];i++){
        if(connections[p->index][i+1]==q->index) return true;
    }
    return false;
}