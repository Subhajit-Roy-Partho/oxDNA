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
    int i=0;
    while(std::getline(topology,line)) {
        if(line.size()==0 || line[0]=='#') continue;
        if(line[0]=='i'){
            std::istringstream body(line);
            body>>temp;
            if(temp=="iP"){
                body>>temp;
                body>>Patches[i][0]>>Patches[i][1]>>Patches[i][2]>>Patches[i][3]>>Patches[i][4];
            }else if(temp=="iS"){
                body>>temp;
                body>>Springs[i][0]>>Springs[i][1]>>Springs[i][2]>>Springs[i][3]>>Springs[i][4];
            }
        }
        if(i>=particleNum) throw oxDNAException("Topology file has more particles than specified in the header");
        std::istringstream body(line);
        // int j=0;
        body>>particles[i]->type>>particles[i]->strand_id>>particleRadius[i];
        auto *pp = dynamic_cast<PSP2Particle *>(particles[i]);
        body>>ParticlePatches[i][0];
        for(int p=0;p<ParticlePatches[i][0];p++){
            body>>ParticlePatches[i][p+1];
            pp->iP.push_back(ParticlePatches[i][p+1]);
        }
        int t=0;
        // while(body.tellg()!=-1){
        //     body>>ParticleSprings[i][t+1];
        //     pp->iS.push_back(ParticleSprings[i][t]);
        //     t++;
        // }
        // ParticleSprings[i][0]=t;

        i++;
    }
}

void PSP2Interaction::check_input_sanity(std::vector<BaseParticle *> &particles) {

}

number PSP2Interaction::pair_interaction(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    return 0;
}

number PSP2Interaction::pair_interaction_bonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    return 0;
}

number PSP2Interaction::pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    return 0;
}