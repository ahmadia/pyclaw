#ifndef __CUDACLAW_MAIN_H__
#define __CUDACLAW_MAIN_H__

#include <math.h>
#include "common_vis.h"
#include "problem_setup.h"
#include "Visualizer2D.h"
#include "GlutInterface.h"
#include "step.h"

// Defined falgs to switch between GPUs for debug or run
#define GPU_RELEASE 0
#define GPU_DEBUG 1

void setupCUDA();

template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute);

void gracefulExit();

#endif