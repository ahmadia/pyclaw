#ifndef __CUDACLAW_H__
#define __CUDACLAW_H__

#include "real.h"

int shallow_water_solver_allocate(int cellsX,
                								  int cellsY,
								                  int ghostCells,
                                  int numStates,
                                  int numWaves,
                                  int numCoeff,
                                  real startX,
                                  real endX,
                                  real startY,
                                  real endY,
                                  real startTime,
                                  real endTime);

int shallow_water_solver_setup (int bc_left,
                                int bc_right,
                                int bc_up,
                                int bc_down,
                                int limiter);

int hyperbolic_solver_2d_step (real dt, real* next_dt);
int hyperbolic_solver_2d_get_qbc (real* qbc);
int hyperbolic_solver_2d_set_qbc (real* qbc);

#endif
