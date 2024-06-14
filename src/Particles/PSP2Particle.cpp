#include "PSP2Particle.h"

void PSP2Particle::add_neighbour(BaseParticle *n){
    if(!has_bond(n)){
        auto *q = static_cast<PSP2Particle *>(n);
        neighbours.push_back(q->index);
        q->neighbours.push_back(index);
        ParticlePair newPair(this,n);
        affected.push_back(newPair);
        n->affected.push_back(newPair);
    }
}

bool PSP2Particle::has_bond(BaseParticle *p){
    if(std::find(neighbours.begin(),neighbours.end(),p->index)!=neighbours.end()){
        return true;
    }else{
        return false;
    }
}