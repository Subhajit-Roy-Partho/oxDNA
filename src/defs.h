/**
 * @file    defs.h
 * @date    13/ott/2009
 * @author  lorenzo
 *
 *
 */

#ifndef DEFS_H_
#define DEFS_H_

#define MAX_EXT_FORCES 15

#include <cmath>
#define M_E        2.71828182845904523536
#define M_LOG2E    1.44269504088896340736
#define M_LOG10E   0.434294481903251827651
#define M_LN2      0.693147180559945309417
#define M_LN10     2.30258509299404568402
#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define M_PI_4     0.785398163397448309616
#define M_1_PI     0.318309886183790671538
#define M_2_PI     0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2    1.41421356237309504880
#define M_SQRT1_2  0.707106781186547524401

double drand48(void) {
    return rand() / (RAND_MAX + 1.0);
}

long int lrand48(void) {
    return rand();
}

long int mrand48(void) {
    return rand() > RAND_MAX / 2 ? rand() : -rand();
}

void srand48(long int seedval) {
    srand(seedval);
}

#define PI 3.141592653589793238462643f
#define SQR(x) ((x) * (x))
#define CUB(x) ((x) * (x) * (x))
#define LRACOS(x) (((x) > 1) ? (number) 0 : ((x) < -1) ? (number) PI : acos(x))

#define CHECK_BOX(my_class, inp) 	std::string box_type ("");\
	if (getInputString(&inp, "box_type", box_type, 0) == KEY_FOUND) {\
		if (box_type.compare("cubic") != 0) \
			throw oxDNAException ("%s only works with cubic box! Aborting", my_class);\
	}\

#define P_A 0
#define P_B 1
#define P_VIRTUAL (NULL)
#define P_INVALID (-1)

#define N_A 0
#define N_G 1
#define N_C 2
#define N_T 3
#define N_DUMMY 4

//Amino Acids Added
#define A_A (5)
#define A_R (6)
#define A_N (7)
#define A_D (8)
#define A_C (9)
#define A_E (10)
#define A_Q (11)
#define A_G (12)
#define A_H (13)
#define A_I (14)
#define A_L (15)
#define A_K (16)
#define A_M (17)
#define A_F (18)
#define A_P (19)
#define A_S (20)
#define A_T (21)
#define A_W (22)
#define A_Y (23)
#define A_V (24)
#define A_DUMMY (25)
#define A_INVALID (26)

#include "model.h"
#include "Utilities/LR_vector.h"
#include "Utilities/LR_matrix.h"
#include "Utilities/Logger.h"
#include "Utilities/parse_input/parse_input.h"

#include <string>
#include <memory>
#include <array>

using uint = uint32_t;
using llint = long long int;
using StressTensor = std::array<number, 6>;

#endif /* DEFS_H_ */
