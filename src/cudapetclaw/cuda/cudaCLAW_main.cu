#define VIS
#include "cudaClaw_main.h"

#include <iostream>

// Defined falgs to switch between GPUs for debug or run
#define GPU_RELEASE 0
#define GPU_DEBUG 1


// Problem Objects
template<class Vis>
Vis* GlutInterface<Vis>::visualizer = NULL;

typedef Visualizer2D<Burger_horizontal, Burger_vertical, limiter_MC, 
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_absorbing>, entropy_fix_Burger_horizontal, entropy_fix_Burger_vertical> Burger_vis;

// This object holds the solver and limiter, and must be generated
typedef Visualizer2D<acoustics_horizontal, acoustics_vertical, limiter_MC, 
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_absorbing>, null_entropy, null_entropy> acoustics_vis;
	//boundaryConditions<BC_left_absorbing, BC_right_absorbing, BC_up_absorbing, BC_down_absorbing> > acoustics_vis;

typedef Visualizer2D<shallow_water_horizontal, shallow_water_vertical, limiter_MC, 
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_absorbing>, entropy_fix_Shallow_Water_horziontal, entropy_fix_Shallow_Water_vertical> shallowWater_vis;

int main(int argc, char** argv)
{
	setupCUDA();

	// Boundary setup
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_absorbing> reflective_conditions;

	BC_left_reflective left;
	BC_right_reflective right;
	BC_up_reflective up;
	BC_down_absorbing down;

	//boundaryConditions<BC_left_absorbing, BC_right_absorbing, BC_up_absorbing, BC_down_absorbing> absorbing_conditions;
	//BC_left_absorbing left;
	//BC_right_absorbing right;
	//BC_up_absorbing up;
	//BC_down_absorbing down;

	reflective_conditions.condition_left = left;
	reflective_conditions.condition_right = right;
	reflective_conditions.condition_up = up;
	reflective_conditions.condition_down = down;

	// Solver setup
	// acoustics
	acoustics_horizontal acoustic_h;
	acoustics_vertical   acoustic_v;
	// shallow water
	shallow_water_horizontal shallow_water_h;
	shallow_water_vertical   shallow_water_v;
	// Burger's solver
	Burger_horizontal	Burger_h;
	Burger_vertical		Burger_v;

	// Limiter setup
	limiter_MC phi;
	limiter_superbee phi1;

	// Entropy fix
	entropy_fix_Shallow_Water_horziontal 	shallow_water_fix_h;
	entropy_fix_Shallow_Water_vertical 		shallow_water_fix_v;
	entropy_fix_Burger_horizontal 			Burger_fix_h;
	entropy_fix_Burger_vertical 			Burger_fix_v;
	null_entropy 							no_entropy;

	int cellsX = 1024;
	int cellsY = 1024;
	real ratio = (real)cellsY/(real)cellsX;

	real simulation_start = 0.0f;
	real simulation_end = 1.0f;

	real snapshotRate = 0;
	bool entropy_fix = false;

	GlutInterface<acoustics_vis>::InitGlut(argc, argv, 512, 512, cellsX, cellsY);	// the last 2 arguments are the resolution of the texture, may be changed inside the method
	//GlutInterface<shallowWater_vis>::InitGlut(argc, argv, 512, 512, cellsX, cellsY);	// the last 2 arguments are the resolution of the texture, may be changed inside the method
	//GlutInterface<Burger_vis>::InitGlut(argc, argv, 512, 512, cellsX, cellsY);	// the last 2 arguments are the resolution of the texture, may be changed inside the method

	//pdeParam problemParam = setup(cellsX, cellsY, -1, 1, -1, ratio, entropy_fix, simulation_start, simulation_end, snapshotRate, radial_plateau);
	//pdeParam problemParam = setup(cellsX, cellsY, -1, 1, -1, ratio, entropy_fix, simulation_start, simulation_end, snapshotRate, radial_plateau/*cone_square/*neg_pos_strip/**/, uniform_coefficients, angles);
	
	real* coeffs = new real[2];
	coeffs[0] = 4.0;
	coeffs[1] = 1.0f;
	pdeParam problemParam = setup(cellsX, cellsY, 0,1,0,ratio, false, 0, 1.0f, 0.0f, /*off_circle_q/*/centered_circle_q/**/, uniform_coefficients, coeffs);
	delete coeffs;

	solvePDE<acoustics_vis>(problemParam, acoustic_h, acoustic_v, phi, reflective_conditions, no_entropy, no_entropy);
	//solvePDE<Burger_vis>(problemParam, Burger_h, Burger_v, phi, reflective_conditions, Burger_fix_h, Burger_fix_v);
	//solvePDE<shallowWater_vis>(problemParam, shallow_water_h, shallow_water_v, phi, reflective_conditions, shallow_water_fix_h, shallow_water_fix_v);
	
	// We never reach this step as the game loop is launched before it
	gracefulExit();
}

template<class Vis, class Solver_h, class Solver_v, class Limiter, class Conditions, class Entropy_h, class Entropy_v>
void solvePDE(pdeParam &params, Solver_h solver_h, Solver_v solver_v, Limiter phi, Conditions conds, Entropy_h ent_fix_h, Entropy_v ent_fix_v)
{
	GlutInterface<Vis>::visualizer->setParam(params);
	GlutInterface<Vis>::visualizer->setBoundaryConditions(conds);
	GlutInterface<Vis>::visualizer->setSolvers(solver_h, solver_v);
	GlutInterface<Vis>::visualizer->setEntropy(ent_fix_h, ent_fix_v);
	GlutInterface<Vis>::visualizer->setLimiter(phi);
	GlutInterface<Vis>::visualizer->initializeDisplay();
	GlutInterface<Vis>::visualizer->launchDisplay();
}

void setupCUDA()
{
	int device = GPU_DEBUG;	//1 for debug 0 for run, chooses the gpu

	// cudaSetDevice and cudaGLSetGLDevice do not make contexts
	// if both choose the same device, the cuda runtime functions
	// will not work properly, so only one of the setter functions
	// must be called, and so cudaGLSetGLDevice chooses the first CUDA device
	cudaError_t errorDevice = cudaSetDevice(device);
	cudaError_t errorGLdevice = cudaGLSetGLDevice(device);

	cudaDeviceProp device_property;
	cudaGetDeviceProperties(&device_property, device);

	// Some error when choosing cache configuration, could be with the order of the call, error code 46-GPU unavailable
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
	delete(GlutInterface<acoustics_vis>::visualizer);
	cudaThreadExit();
	exit(0);
}