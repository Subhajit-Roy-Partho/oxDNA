/*
 * EdgeList.h
 *
 * Verlet-skin neighbour list for CPU MD/MC simulations.
 * Maintains per-particle half-lists (q->index < p->index, non-bonded only)
 * for get_neigh_list(). get_potential_interactions() is inherited from
 * BaseList and correctly includes bonded and non-bonded pairs.
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
    virtual void change_box();
};

#endif /* EDGELIST_H_ */
