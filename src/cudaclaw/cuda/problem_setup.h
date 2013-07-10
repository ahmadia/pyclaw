#ifndef __PROBLEM_SETUP_H__
#define __PROBLEM_SETUP_H__

#define NUMSTATES 3
#define NUMWAVES 3
#define NUMCOEFF 1

#include <math.h>

#include "common.h"

// State initial value functions
real* mid_Gaussian_q(pdeParam &);
real* circle_q(pdeParam &param);
real* centered_circle_q(pdeParam &param);
real* off_circle_q(pdeParam &param);

real* radial_plateau(pdeParam &);
real* separating_streams(pdeParam &);
real* dam_break(pdeParam &param);

real* neg_pos_strip(pdeParam &);
real* cone_square(pdeParam &);

// Coefficient value functions
real* uniform_coefficients(pdeParam &, real* u);
real* some_function_coefficients(pdeParam &);

// Problem Setup
pdeParam setup(int cellsX, int cellsy, real x_start, real x_end, real y_start, real y_end, bool entropy_fix, real time_start, real time_end, real snapshotRate, real*(*init_q_function)(pdeParam &), real*(*init_coeff_function)(pdeParam &, real* u), real* coeff_input);
pdeParam setup(int cellsX, int cellsy, real x_start, real x_end, real y_start, real y_end, bool entropy_fix, real time_start, real time_end, real snapshotRate, real*(*init_q_function)(pdeParam &), real*(*init_coeff_function)(pdeParam &));
pdeParam setup(int cellsX, int cellsy, real x_start, real x_end, real y_start, real y_end, bool entropy_fix, real time_start, real time_end, real snapshotRate, real*(*init_q_function)(pdeParam &));

// OBSOLETE - DEPRECATED
//pdeParam setupAcoustics(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void(*init_q_function)(pdeParam &), void(*init_coeff_function)(pdeParam &, real* u));
//pdeParam setupAcoustics(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void(*init_q_function)(pdeParam &), void(*init_coeff_function)(pdeParam &));
//pdeParam setupShallowWater(real x_start, real x_end, real y_start, real y_end, bool entropy_fix, real time_start, real time_end, real snapshotRate, void(*init_q_function)(pdeParam &));
//pdeParam setupBurger(real x_start, real x_end, real y_start, real y_end, bool entropy_fix, real time_start, real time_end, real snapshotRate, void(*init_q_function)(pdeParam &));

//
void setUpProblem(int argc, char** argv);

#endif
