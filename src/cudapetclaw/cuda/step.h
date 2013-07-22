#ifndef __STEP_H__
#define __STEP_H__

#include "boundary_conditions.h"
#include "fused_Riemann_Limiter.h"

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
    limited_Riemann_Update(param, Riemann_pointwise_solver_h, Riemann_pointwise_solver_v, limiter_phi, entropy_fix_h, entropy_fix_v);
    cudaMemcpy(&dt, param.dt_used, sizeof(real), cudaMemcpyDeviceToHost);

    return dt;
}

#endif
