/*
 * EdgeList.h
 *
 * Efficient flat edge-list neighbour list for CPU MD simulations.
 * Stores all unique (p, q) pairs with p->index > q->index in a flat
 * contiguous array, enabling cache-friendly iteration over all pairs.
 *
 * Compatible with existing BaseList interface: get_neigh_list() works for MC,
 * while get_potential_interactions() returns the pre-built flat edge array
 * in O(1) instead of O(N * avg_neigh) as in the base class.
 */

#ifndef EDGELIST_H_
#define EDGELIST_H_

#include "Cells.h"

/**
 * @brief Flat edge-list neighbour structure with Verlet skin for lazy updates.
 *
 * @verbatim
verlet_skin = <float> (Verlet skin width; list rebuilt when any particle
                       displaces more than this distance from its last-update position.)
@endverbatim
 */
class EdgeList: public BaseList {
protected:
    // flat list of all unique pairs (p->index > q->index)
    std::vector<ParticlePair> _edges;
    // per-particle half-neighbour lists for MC/get_neigh_list compatibility
    std::vector<std::vector<BaseParticle *>> _neigh_lists;
    // positions at last global update, used for skin check
    std::vector<LR_vector> _list_poss;

    number _skin;
    number _sqr_skin;
    bool _updated;

    Cells _cells;

    void _rebuild_lists();

public:
    EdgeList(std::vector<BaseParticle *> &ps, BaseBox *box);
    EdgeList() = delete;
    virtual ~EdgeList();

    virtual void get_settings(input_file &inp);
    virtual void init(number rcut);

    virtual bool is_updated();
    virtual void single_update(BaseParticle *p);
    virtual void global_update(bool force_update = false);
    virtual std::vector<BaseParticle *> get_neigh_list(BaseParticle *p);
    virtual std::vector<BaseParticle *> get_complete_neigh_list(BaseParticle *p);
    virtual std::vector<ParticlePair> get_potential_interactions();
    virtual void change_box();
};

#endif /* EDGELIST_H_ */
