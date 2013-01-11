#include "cudaclaw.h"
#include "common.h"
#include "problem_setup.h"

#include "boundary_conditions.h"
#include "fused_Riemann_Limiter.h"

#define GPU_RELEASE 0
#define GPU_DEBUG 1

void setupCUDA();

template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute);


pdeParam *param;
boundaryConditions<BC_left_absorbing, 
				   BC_right_absorbing, 
				   BC_up_absorbing, 
				   BC_down_absorbing> bc;

shallow_water_horizontal shallow_water_h;
shallow_water_vertical   shallow_water_v;
limiter_MC phi_mc;
real* cpu_q;
size_t qbc_size;
cudaError_t err;

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
                                  real endTime) 
{
	param = new pdeParam(cellsX,
      				     cellsY,
      				     ghostCells,
      				     numStates,
      				     numWaves,
      				     numCoeff,
      				     startX,
      				     endX,
      				     startY,
      				     endY,
      				     startTime,
      				     endTime);

	qbc_size = param->cellsX*param->cellsY*param->numStates*sizeof(real);

	return 0;
}

int shallow_water_solver_setup (int bc_left, 
                                int bc_right, 
                                int bc_up, 
                                int bc_down, 
                                int limiter
                                ) 
{

	setupCUDA();	


	// the boundary conditions and limiter are currently hard-coded to 
	// absorbing, mc

	BC_left_absorbing left;
	BC_right_absorbing right;
	BC_up_absorbing up;
	BC_down_absorbing down;

   	bc.condition_left = left;
	bc.condition_right = right;
	bc.condition_up = up;
	bc.condition_down = down;

	return 0;
}
    
int hyperbolic_solver_2d_step (real dt, real* next_dt) 
{

	setBoundaryConditions(*param, bc);
	limited_Riemann_Update(*param, dt, 
						   shallow_water_h, 
						   shallow_water_v, 
						   phi_mc);
	err = cudaMemcpy(next_dt, param->waveSpeedsX, sizeof(real), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
    	return err;
	}	
	return 0;
}

int hyperbolic_solver_2d_get_qbc (real* qbc) 
{
    err = cudaMemcpy(qbc, param->qNew, qbc_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
    	return err;
	}	
	return 0;
}

int hyperbolic_solver_2d_set_qbc (real* qbc) 
{
    err = cudaMemcpy(param->qNew, qbc, qbc_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
    	return err;
	}
	return 0;	
}

void setupCUDA()
{
	int device = GPU_RELEASE;	//1 for debug 0 for run, chooses the gpu

	cudaError_t errorDevice = cudaSetDevice(device);

	cudaDeviceProp device_property;
	cudaGetDeviceProperties(&device_property, device);

	// Some error when choosing cache configuration, could be with the order of the call, 
	if (device_property.major < 2)
		// cache-shared memory configuring not possible, no cache
		printf("Cache configuration not possible\n");
	else
	{
		//cudaError_t errorCachePref1 = cudaFuncSetCacheConfig("fused_Riemann_limiter_horizontal_update_kernel", cudaFuncCachePreferShared);
		//cudaError_t errorCachePref2 = cudaFuncSetCacheConfig("fused_Riemann_limiter_vertical_update_kernel", cudaFuncCachePreferShared);
		//printf("Cache configuration done, config1: %i, config2: %i\n",errorCachePref1,errorCachePref2);
	}
}
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
							 int device)
{
	// Credit to Nvidia GPU computing SDK, deviceQuery project.
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if( CUDA_SUCCESS != error)
	{
		printf("cuSafeCallNoSync() Driver API error = %04d\n", error);
        exit(-1);
    }
}
