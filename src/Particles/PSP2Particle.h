#ifndef PSP2Particle_H
#define PSP2Particle_H

#include "BaseParticle.h"
#include <algorithm>

class PSP2Particle: public BaseParticle{
public:
    std::vector<int> iS,iP,neighbours;
    virtual void add_neighbour(BaseParticle *n);
    virtual bool has_bond(BaseParticle *p);
    virtual bool is_rigid_body(){ //without this a1,a3 won't be updated
        return true;
    };
};


#endif