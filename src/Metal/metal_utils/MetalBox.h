/**
 * @file    MetalBox.h
 * @brief   Metal GPU simulation box
 *
 * Metal equivalent of CUDABox
 */

#ifndef METALBOX_H_
#define METALBOX_H_

#include "../metal_defs.h"

/**
 * @brief Simulation box for Metal GPU
 */
struct MetalBox {
    m_number box_sides[3];
    m_number inv_sides[3];

    MetalBox() {
        for(int i = 0; i < 3; i++) {
            box_sides[i] = 0;
            inv_sides[i] = 0;
        }
    }

    void set_sides(m_number x, m_number y, m_number z) {
        box_sides[0] = x;
        box_sides[1] = y;
        box_sides[2] = z;
        inv_sides[0] = 1.0 / x;
        inv_sides[1] = 1.0 / y;
        inv_sides[2] = 1.0 / z;
    }
};

#endif /* METALBOX_H_ */
