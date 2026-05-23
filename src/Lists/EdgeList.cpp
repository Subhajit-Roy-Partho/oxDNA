/*
 * EdgeList.cpp
 *
 * Flat edge-list implementation. Maintains per-particle half-lists for MC
 * compatibility. Uses a Verlet skin so cells are only consulted when particles
 * have displaced by more than _skin since the last update.
 */

#include "EdgeList.h"

EdgeList::EdgeList(std::vector<BaseParticle *> &ps, BaseBox *box) :
        BaseList(ps, box),
        _skin(0.),
        _sqr_skin(0.),
        _updated(false),
        _cells(ps, box) {
}

EdgeList::~EdgeList() {
}

void EdgeList::get_settings(input_file &inp) {
    BaseList::get_settings(inp);
    _cells.get_settings(inp);

    getInputNumber(&inp, "verlet_skin", &_skin, 1);
    _sqr_skin = SQR(_skin);

    if(this->_is_MC) {
        float delta_t = 0.f;
        getInputFloat(&inp, "delta_translation", &delta_t, 0);
        if(delta_t > 0.f && delta_t * sqrt(3) > _skin) {
            throw oxDNAException("verlet_skin must be > delta_translation times sqrt(3) (the maximum displacement)");
        }
    }
}

void EdgeList::init(number rcut) {
    rcut += 2 * _skin;
    BaseList::init(rcut);

    _neigh_lists.resize(_particles.size());
    _list_poss.resize(_particles.size(), LR_vector(0, 0, 0));

    _cells.init(rcut);
    global_update(true);
}

bool EdgeList::is_updated() {
    return _updated;
}

void EdgeList::single_update(BaseParticle *p) {
    _cells.single_update(p);
    if(_list_poss[p->index].sqr_distance(p->pos) > _sqr_skin) {
        _updated = false;
    }
}

void EdgeList::_rebuild_lists() {
    for(auto &nl : _neigh_lists) nl.clear();

    for(uint i = 0; i < _particles.size(); i++) {
        BaseParticle *p = _particles[i];
        _neigh_lists[p->index] = _cells.get_neigh_list(p);
        _list_poss[p->index] = p->pos;
    }

    _updated = true;
}

void EdgeList::global_update(bool force_update) {
    if(!_cells.is_updated() || force_update) {
        _cells.global_update();
    }
    _rebuild_lists();
}

std::vector<BaseParticle *> EdgeList::get_neigh_list(BaseParticle *p) {
    return _neigh_lists[p->index];
}

std::vector<BaseParticle *> EdgeList::get_complete_neigh_list(BaseParticle *p) {
    return _cells.get_complete_neigh_list(p);
}

void EdgeList::change_box() {
    LR_vector new_sides = this->_box->box_sides();
    number fx = new_sides.x / this->_box_sides.x;
    number fy = new_sides.y / this->_box_sides.y;
    number fz = new_sides.z / this->_box_sides.z;

    for(uint i = 0; i < _particles.size(); i++) {
        BaseParticle *p = _particles[i];
        _list_poss[p->index].x *= fx;
        _list_poss[p->index].y *= fy;
        _list_poss[p->index].z *= fz;
        if(_list_poss[p->index].sqr_distance(p->pos) > _sqr_skin) {
            _updated = false;
        }
    }

    _cells.change_box();
    BaseList::change_box();
}
