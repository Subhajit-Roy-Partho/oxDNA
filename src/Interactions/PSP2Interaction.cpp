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
        body>>particles[i]->type>>particleStrand[i]>>particleRadius[i]>>particles[i]->mass;
        particles[i]->invmass=1/particles[i]->mass;
        particles[i]->mr2=particles[i]->mass*particleRadius[i]*particleRadius[i];
        particles[i]->invmr2=1/particles[i]->mr2;
        auto *pp = dynamic_cast<PSP2Particle *>(particles[i]);
        pp->strand_id = particleStrand[i];
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
        // for(int l=0;l< pp->neighbours.size();l++){
        //    std::cout<<pp->index<<" "<<pp->neighbours[l]<<"\t";
        // }
        // std::cout<<"\n";
        i++;
    }

    _rcut=12;
    _sqr_rcut = SQR(_rcut);

    populateInvSprings();


    // Debug Printout

    // for(i=0;i<particleNum;i++){
    //     for(int j=0;j<connections[i][0];j++){
    //         std::cout<<connections[i][j+1]<<" "<<ParticleSprings[i][j]<<" "<<invParticleSprings[i][j]<<"\t";
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

    // number k,r0;
    // LR_vector rp,rq;
    // int next=returnKro(0,1,&k,&r0,&rp,&rq);
    // std::cout<<k<<"\t"<<r0<<"\t rp = "<<rp.x<<" "<<rp.y<<" "<<rp.z<<"\t rq = "<<rq.x<<" "<<rq.y<<" "<<rq.z<<"\n";
    
    // for(int i=0;i<particleNum;i++){
    //     for(int j=0;j<connections[i][0];j++){
    //         std::cout<<SpringR0[i][j]<<"\t";
    //     }
    //     std::cout<<"\n";
    // }
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
    number energy = 0;
    if(p->strand_id==q->strand_id) return energy;
    // std::cout<<"called"<<std::endl;
    if(compute_r) _computed_r = _box->min_image(p->pos, q->pos);
    // energy += exeVol(p,q,compute_r,update_forces);
    // energy+=simplePatch(p,q,compute_r,update_forces);
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
    number energy =0;
    rmod = _computed_r.module();
    if(rmod-(particleRadius[p->index]+particleRadius[q->index])>patchyRcut) return 0;
    if(ParticlePatches[q->index][0]==0) return 0;
    for(int pi=0;pi<ParticlePatches[p->index][0];pi++){
        LR_vector ppatch= p->orientation*returnPatch(p->index,pi);
        for(int qi=0;qi<ParticlePatches[q->index][0];qi++){
            if(Patches[ParticlePatches[p->index][pi+1]][0]+Patches[ParticlePatches[q->index][qi+1]][0]!=0) continue; //color check
            LR_vector qpatch= q->orientation*returnPatch(q->index,qi);
            LR_vector r = _computed_r - ppatch + qpatch;
            number dist = r.norm();
            if(dist<patchyRcut2){
                number K = Patches[ParticlePatches[p->index][pi+1]][1]+Patches[ParticlePatches[q->index][qi+1]][1]; //total strength
                number r2b2=dist*patchyAlphaB2;
                number r8b10 = r2b2*r2b2*r2b2*r2b2*patchyAlphaB2;
                energy = -1.f*exp(-1.f*r8b10*dist)*K;
                if(update_forces){
                    LR_vector force = r*(10*energy*r8b10);
                    p->force-=force;
                    q->force+=force;
                    p->torque -= p->orientationT*ppatch.cross(force);
                    q->torque += q->orientationT*qpatch.cross(force);
                }
            }
        }
    }

    return energy;
}

//Spring Function

number PSP2Interaction::torqueSpring(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    number energy =0;
    for(int i=0;i<maxSprings;i++){
        number k,r0;
        LR_vector SpringP,SpringQ;
        int next;
        if(i==0){
            next=returnKro(p->index,q->index,&k,&r0,&SpringP,&SpringQ);
            // std::cout<<"p = "<<p->index<<"q = "<<q->index<<"\t"<<next<<"\n";
        }else{
            next=returnKro(p->index,q->index,&k,&r0,&SpringP,&SpringQ,next);
        }
        LR_vector intCenterP = p->orientation*SpringP;
        LR_vector intCenterQ = q->orientation*SpringQ; //Here
        LR_vector r = _computed_r - intCenterP + intCenterQ;
        number modr = r.module();
        number dist = modr - r0;
        // energy += 0.25*k*SQR(dist);

        // LINEAR SPRING
        number modr2 = _computed_r.module();
        number dist2 = modr2 - SpringR0[p->index][next-1];
        energy += 0.25*k*SQR(dist2)*springMultiplier;
        /////////////////

        if(update_forces){
            LR_vector force={0,0,0};
            // LR_vector force = -k*dist*(r/modr);

            // p->torque -= p->orientationT*intCenterP.cross(force);
            // q->torque += q->orientationT*intCenterQ.cross(force);

            force -= k*dist2*(_computed_r/modr2)*springMultiplier;  //LINEAR SPRING
            p->force-=force;
            q->force+=force;
        }
    }
    return energy;
}

// number PSP2Interaction::Spring(BaseInteraction *p, BaseInteraction *q, bool compute_r, bool update_forces) {
//     LR_vector r = _computed_r;
//     number modr = r.module();
//     number dist = modr - SpringR0[]
// }
//Common Functions

bool PSP2Interaction::bonded(BaseParticle *p, BaseParticle *q) {
    for(int i=0;i<connections[p->index][0];i++){
        if(connections[p->index][i+1]==q->index) return true;
    }
    return false;
}

int PSP2Interaction::returnKro(int p, int q, number *k, number *r0, LR_vector *rp,LR_vector *rq) {
    
    for(int i=0;i<connections[p][0];i++){
        if(connections[p][i+1]==q){
            *k = Springs[ParticleSprings[p][i]][0];
            *r0 = Springs[ParticleSprings[p][i]][1];
            *rp = (LR_vector){Springs[ParticleSprings[p][i]][2],Springs[ParticleSprings[p][i]][3],Springs[ParticleSprings[p][i]][4]};
            *rq = (LR_vector){Springs[invParticleSprings[p][i]][2],Springs[invParticleSprings[p][i]][3],Springs[invParticleSprings[p][i]][4]};
            return i+1;
        }
    }
    return -1;
}

int PSP2Interaction::returnKro(int p, int q, number *k, number *r0, LR_vector *rp, LR_vector *rq, int next) {
    if(connections[p][next+1]==q){
        *k = Springs[ParticleSprings[p][next]][0];
        *r0 = Springs[ParticleSprings[p][next]][1];
        *rp = (LR_vector){Springs[ParticleSprings[p][next]][2],Springs[ParticleSprings[p][next]][3],Springs[ParticleSprings[p][next]][4]};
        *rq = (LR_vector){Springs[invParticleSprings[p][next]][2],Springs[invParticleSprings[p][next]][3],Springs[invParticleSprings[p][next]][4]};
        return next+1;
    }
    return -1;
}

LR_vector PSP2Interaction::returnPatch(int p,int i) {
    return (LR_vector){Patches[ParticlePatches[p][i+1]][2],Patches[ParticlePatches[p][i+1]][3],Patches[ParticlePatches[p][i+1]][4]};
}

//Extra Topology Functions

void PSP2Interaction::populateInvSprings() {
    for(int p=0;p<particleNum;p++){
        for(int j=0;j<connections[p][0];j++){
            int q = connections[p][j+1];
            SpringR0[p][j]=Springs[ParticleSprings[p][j]][1]+particleRadius[p]+particleRadius[q];
            for(int i=0;i<connections[q][0];i++){
                if(connections[q][i+1]==p){
                    invParticleSprings[p][j]=ParticleSprings[q][i];
                    break;
                }
            }
        }
    }
}