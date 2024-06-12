#include "PSP2Particle.h"

void PSP2Particle::add_neighbour(BaseParticle *n){
    ParticlePair newPair(this,n);
    affected.push_back(newPair);
    n->affected.push_back(newPair);
}