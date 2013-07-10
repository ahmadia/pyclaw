#include "cudaClaw_main_noVis.h"

#include <iostream>

// Defined falgs to switch between GPUs for debug or run
#define GPU_RELEASE 0
#define GPU_DEBUG 1

int main(int argc, char** argv)
{
	setupCUDA();

	// Boundary setup
	//boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_reflective> reflective_conditions;
	//boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_absorbing> semi_reflective_conditions;

	//BC_left_reflective left;
	//BC_right_reflective right;
	//BC_up_reflective up;
	//BC_down_reflective down;

	//reflective_conditions.condition_left = left;
	//reflective_conditions.condition_right = right;
	//reflective_conditions.condition_up = up;
	//reflective_conditions.condition_down = down;

	boundaryConditions<BC_left_absorbing, BC_right_absorbing, BC_up_absorbing, BC_down_absorbing> absorbing_conditions;
	BC_left_absorbing left;
	BC_right_absorbing right;
	BC_up_absorbing up;
	BC_down_absorbing down;

	absorbing_conditions.condition_left = left;
	absorbing_conditions.condition_right = right;
	absorbing_conditions.condition_up = up;
	absorbing_conditions.condition_down = down;

	// Solver setup
	// acoustics
	acoustics_horizontal acoustic_h;
	acoustics_vertical   acoustic_v;
	// shallow water
	shallow_water_horizontal shallow_water_h;
	shallow_water_vertical   shallow_water_v;

	// Limiter setup
	limiter_MC phi;
	limiter_superbee phi1;

	// Entropy fix
	entropy_fix_Shallow_Water_horziontal ent_fix_shallow_water_h;
	entropy_fix_Shallow_Water_vertical   ent_fix_shallow_water_v;
	null_entropy no_entropy;

	int cellsX = 128;
	int cellsY = 128;
	real ratio = (real)cellsY/(real)cellsX;

	real simulation_start_time = 0.0f;
	real simulation_end_time = 1.0f;

	real snapshotRate = 0.00f;
	bool entropy_fix = true;

	pdeParam problemParam = setup(cellsX, cellsY, -1, 1, -1, 1, entropy_fix, simulation_start_time, simulation_end_time, snapshotRate, radial_plateau);
	//real* coeffs = new real[2];
	//coeffs[0] = 4.0f;
	//coeffs[1] = 1.0f;

	//pdeParam problemParam = setup(cellsX, cellsY, 0, 1, 0, 1, entropy_fix, simulation_start_time, simulation_end_time, snapshotRate, centered_circle_q, uniform_coefficients, coeffs);
	//delete coeffs;

	//solvePDE(problemParam, acoustic_h, acoustic_v, phi, reflective_conditions, no_entropy, no_entropy);
	solvePDE(problemParam, shallow_water_h, shallow_water_v, phi, absorbing_conditions, ent_fix_shallow_water_h, ent_fix_shallow_water_v);

	problemParam.clean();
	gracefulExit();
}

template<class Solver_h, class Solver_v, class Limiter, class Conditions, class Entropy_h, class Entropy_v>
void solvePDE(pdeParam &params, Solver_h solver_h, Solver_v solver_v, Limiter phi, Conditions conds, Entropy_h ent_fix_h, Entropy_v ent_fix_v)
{
	// This single step seems necessary for the data to show
	step(params, solver_h, solver_v, phi, conds, ent_fix_h, ent_fix_v);

	if( params.snapshots )
		params.takeSnapshot(0, "pde data");	// take initial state snapshot

	int snap_number = 1;
	
	real simulationTime = 0.0f;
	real simulationStepTime = 0.0f;
	real simulationTimeInterval = 0.0f;
	while (simulationTime < params.endTime)
	{
		simulationStepTime = step(params, solver_h, solver_v, phi, conds, ent_fix_h, ent_fix_v);
		printf("Simulation Time is: %fs\n", simulationTime);

		simulationTime += simulationStepTime;
		simulationTimeInterval += simulationStepTime;
		if (params.snapshots && simulationTimeInterval > params.snapshotTimeInterval)
		{
			params.takeSnapshot(snap_number, "pde data");
			simulationTimeInterval = 0.0f;
			snap_number++;
		}
	}
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
void gracefulExit()
{
	cudaThreadExit();
	exit(0);
}
