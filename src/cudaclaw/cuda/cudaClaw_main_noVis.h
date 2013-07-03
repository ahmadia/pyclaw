#ifndef __CUDACLAW_MAIN_NOVIS_H__
#define __CUDACLAW_MAIN_NOVIS_H__

#include <math.h>
#include "common.h"
#include "problem_setup.h"
#include "step.h"

// Defined falgs to switch between GPUs for debug or run
#define GPU_RELEASE 0
#define GPU_DEBUG 1

void setupCUDA();

template<class Solver_h, class Solver_v, class Limiter, class Conditions, class Entropy_h, class Entropy_v>
void solvePDE(pdeParam &, Solver_h, Solver_v, Limiter, Conditions, Entropy_h, Entropy_v);

template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute);

void gracefulExit();

#endif