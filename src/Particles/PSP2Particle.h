#ifndef PSP2Particle_H
#define PSP2Particle_H

#include "BaseParticle.h"
#include <algorithm>

class PSP2Particle: public BaseParticle{
public:
    std::vector<int> iS,iP;
    virtual void add_neighbour(BaseParticle *n);
};


#endif