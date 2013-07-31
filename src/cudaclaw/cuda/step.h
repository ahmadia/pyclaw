#ifndef __STEP_H__
#define __STEP_H__

#include "boundary_conditions.h"
#include "fused_Riemann_Limiter.h"

real get_horizontal_absMax_speed(pdeParam &param)
{
	real horizontal_speed = 0.0f;
	cudaMemcpy(&horizontal_speed, param.waveSpeedsX, sizeof(real), cudaMemcpyDeviceToHost);
	return horizontal_speed;
}
real get_vertical_absMax_speed(pdeParam &param)
{
	real vertical_speed = 0.0f;
	cudaMemcpy(&vertical_speed, param.waveSpeedsY, sizeof(real), cudaMemcpyDeviceToHost);
	return vertical_speed;
}
void time_step_adjustment(pdeParam &param, real &dt, bool &revert)
{
	// Function to check the necessary data to decide a time step and validity of lastest step
}

template <class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
    real step(pdeParam &param,			        // Problem parameters
	      Riemann_h Riemann_pointwise_solver_h,	// Riemann problem solver for vertical interfaces, device function
	      Riemann_v Riemann_pointwise_solver_v,	// Riemann problem solver for horizontal interfaces device function
	      Limiter limiter_phi,			// limiter device function called from a fused Riemann-Limiter function,
		                                        // takes the Riemann solver as parameter along other parameters (waves, and others?)
	      BCS boundary_conditions,		        // Boundary conditions put in one object
	      Ent_h entropy_fix_h,		        // Entropy fix for horizontal
	      Ent_v entropy_fix_v)			// Entropy fix for vertical

{
    real dt;

    setBoundaryConditions(param, boundary_conditions);
    limited_Riemann_Update(param, Riemann_pointwise_solver_h, Riemann_pointwise_solver_v, limiter_phi, entropy_fix_h, entropy_fix_v, 0.0001f, false);
    cudaMemcpy(&dt, param.dt_used, sizeof(real), cudaMemcpyDeviceToHost);

    return dt;
}

#endif
