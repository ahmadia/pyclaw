#ifndef __FUSED_RIEMANN_LIMITER_H__
#define __FUSED_RIEMANN_LIMITER_H__

#include "fused_Riemann_Limiter_headers.h"
#include "entropy_fix_header.h"
#include "auxiliary_device_functions.h"
#include "reduce_Max.h"

// GETTERS AND SETTER OF SHARED MEMORY //

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////   Fused Solvers   //////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ real shared_elem[];
template <const int numStates, const int numWaves, const int numCoeff, const unsigned int blockSize, class Riemann, class Limiter, class Ent>
__global__ void Riemann_horizontal_kernel(pdeParam param, Riemann Riemann_solver_h, Limiter phi, Ent entropy_fixer)
{
	// Every threads gets a unique interface
	// a thread at (row, col) treats the interface between cells
	// (row,col)|(row,col+1)
	//          /\
	//		thread (row,col)
	int col = threadIdx.x + blockIdx.x*blockDim.x - 3*blockIdx.x;
	int row = threadIdx.y + blockIdx.y*blockDim.y;

	real* wavesX		= &shared_elem[0];
	real* waveSpeedsX	= &shared_elem[blockSize*numWaves*numStates];

	real dt = *param.dt;

	real apdq[numStates];
	real amdq[numStates];

	real leftCell[numStates];	real leftCoeff[numCoeff];
	real rightCell[numStates];  real rightCoeff[numCoeff];

	if (blockIdx.x > gridDim.x-3 || blockIdx.y > gridDim.y-2)
	{
		#pragma unroll
		for (int i = 0; i < numWaves; i++)
			setSharedSpeed(wavesX, threadIdx.y, threadIdx.x, i, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX, (real)0.0f);
	}

	bool grid_valid = row < param.cellsY && col < param.cellsX-1;

	// Riemann solver
	if (grid_valid)	// if there are 512 cells in X 511 threads are in use
	{
		#pragma unroll
		for (int i = 0; i < numStates; i++)
		{
			leftCell[i] = param.getElement_qNew(row,col,i);
			rightCell[i] = param.getElement_qNew(row,col+1,i);
		}
		#pragma unroll
		for (int i = 0; i < numCoeff; i++)
		{
			leftCoeff[i] = param.getElement_coeff(row,col,i);
			rightCoeff[i] = param.getElement_coeff(row,col+1,i);
		}
		Riemann_solver_h(	leftCell,			// input comes from global memory
							rightCell,			//
							leftCoeff,			//
							rightCoeff,			//
							threadIdx.y,		//
							threadIdx.x,		//
							numStates,			//
							numWaves,			//
							wavesX,				//
							waveSpeedsX);
	}

	grid_valid = grid_valid && ( 0 < threadIdx.x && threadIdx.x < HORIZONTAL_K_BLOCKSIZEX-1 );

	if (grid_valid)
	{
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			amdq[k] = 0;
			apdq[k] = 0;
		}
		if(param.entropy_fix)
		{
			entropy_fixing_h<numStates, numWaves>(entropy_fixer, wavesX, waveSpeedsX, leftCell, rightCell, leftCoeff, rightCoeff, amdq, apdq, threadIdx.y, threadIdx.x);

			//entropy_fix_Burger_horizontal my_entropy_fixer;
			//entropy_fixing_h<numStates, numWaves, entropy_fix_Burger_horizontal>(my_entropy_fixer, wavesX, waveSpeedsX, leftCell, rightCell, leftCoeff, rightCoeff, amdq, apdq, threadIdx.y, threadIdx.x);

			// Straight for Burger's equation's entropy fix
			/*real waveSpeed = getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, 0, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);;

			if(waveSpeed < 0.0f)
				amdq[0] = waveSpeed * getSharedWave(wavesX, threadIdx.y, threadIdx.x, 0, 0, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			else
				apdq[0] = waveSpeed * getSharedWave(wavesX, threadIdx.y, threadIdx.x, 0, 0, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

			if(rightCell[0] < 0 && leftCell[0] > 0)
			{
				real theta = 0.0f;
				amdq[0] = -0.5f*cos(leftCoeff[0])*leftCell[0]*leftCell[0];
				apdq[0] =  0.5f*cos(leftCoeff[0])*rightCell[0]*rightCell[0];
			}*/
			//if (param.second_order)
			{
				second_order_update_horizontal<numStates, numWaves, HORIZONTAL_K_BLOCKSIZEX, HORIZONTAL_K_BLOCKSIZEY>(param, wavesX, waveSpeedsX, amdq, apdq, threadIdx.y, threadIdx.x, dt, phi);
				/*/#pragma unroll
				for (int w = 0; w < numWaves; w++)
				{
					real waveSpeed =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

					if (waveSpeed < (real)0.0f)
					{
						real limiting_factor = limiting_shared_h_l<numStates, numWaves, HORIZONTAL_K_BLOCKSIZEX, HORIZONTAL_K_BLOCKSIZEY>(phi, wavesX, threadIdx.y, threadIdx.x, w);

						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
							amdq[k] +=	//waveSpeed * wave_state
									+
										-(real)0.5f*waveSpeed*(1+(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
						}
					}
					else
					{
						real limiting_factor = limiting_shared_h_r<numStates, numWaves, HORIZONTAL_K_BLOCKSIZEX, HORIZONTAL_K_BLOCKSIZEY>(phi, wavesX, threadIdx.y, threadIdx.x, w);

						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
							apdq[k] +=	//waveSpeed * wave_state
									-
										(real)0.5f*waveSpeed*(1-(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
						}
					}
				}/**/
			}
		}
		else
		{
			if (!param.second_order)	// simple first order scheme
			{
				first_order_update<numStates, numWaves, HORIZONTAL_K_BLOCKSIZEX, HORIZONTAL_K_BLOCKSIZEY> (wavesX, waveSpeedsX, amdq, apdq, threadIdx.y, threadIdx.x);
				/*/#pragma unroll
				for (int w = 0; w < numWaves; w++)
				{
					real waveSpeed =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
					if (waveSpeed < (real)0.0f)
					{
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
							amdq[k] +=	waveSpeed * wave_state;
						}
					}
					else
					{
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
							apdq[k] +=	waveSpeed * wave_state;
						}
					}
				}/**/
			}
			else
			{
				first_second_order_update_horizontal<numStates, numWaves, HORIZONTAL_K_BLOCKSIZEX, HORIZONTAL_K_BLOCKSIZEY> (param, wavesX, waveSpeedsX, amdq, apdq, threadIdx.y, threadIdx.x, dt, phi);
				/*/#pragma unroll
				for (int w = 0; w < numWaves; w++)
				{
					real waveSpeed =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

					if (waveSpeed < (real)0.0f)
					{
						real limiting_factor = limiting_shared_h_l<numStates, numWaves, HORIZONTAL_K_BLOCKSIZEX, HORIZONTAL_K_BLOCKSIZEY>(phi, wavesX, threadIdx.y, threadIdx.x, w);

						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
							amdq[k] +=	waveSpeed * wave_state
									+
										-(real)0.5f*waveSpeed*(1+(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
						}
					}
					else
					{
						real limiting_factor = limiting_shared_h_r<numStates, numWaves, HORIZONTAL_K_BLOCKSIZEX, HORIZONTAL_K_BLOCKSIZEY>(phi, wavesX, threadIdx.y, threadIdx.x, w);

						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
							apdq[k] +=	waveSpeed * wave_state
									-
										(real)0.5f*waveSpeed*(1-(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
						}
					}
				}/**/
			}
		}
	}
	__syncthreads();

	// write the apdq to shared memory
	#pragma unroll
	for(int k = 0; k < numStates; k++)
	{
		setSharedWave(wavesX, threadIdx.y, threadIdx.x, 0, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX, apdq[k]);
	}

	// Local Reduce over Wave Speeds
	// Stage 1
	// Bringing down the number of elements to compare to block size
	int tid = threadIdx.x + threadIdx.y*HORIZONTAL_K_BLOCKSIZEX;
	waveSpeedsX[tid] = fabs(waveSpeedsX[tid]);
	#pragma unroll
	for (int i = 1; i < numWaves; i++)
	{
		// blockDim.x * blockDim.y is usally a multiple of 32, so there should be no bank conflicts.
		waveSpeedsX[tid] = fmax(waveSpeedsX[tid], fabs(waveSpeedsX[tid + i*blockSize]));
	}
	__syncthreads();	// unmovable syncthreads

	// Stage 2
	// Reducing over block size elements
	// use knowledge about your own block size:
	// I am assuming blocks to be of size no more than 512 will be used for the horizontal direction
	// There is a potential for a very subtle bug in the implementation below:
	// if the thread block has size non poer of 2, then there would be 2 threads reading/writing
	// from the same location, for example if the block size is 135, threads [0-67] will be active
	// and thread 0 will operate on (0,67) and thread 67 will operate on (67, 135). This can cause
	// an issue IF the SM somehow reads an unfinished write to the shared memory location. The
	// read data would be junk and could potentially hinder the simulation, either by a crash, or
	// behaving like a large number (behaving as a small number is no problem as this is a max reduce)
	// which could slow down the simulation. In any case block size should be multiples of 32 at least,
	// and never odd numbers. Also if the block size is between 32 and 64 warp reduce might access
	// off limit data. Rule of thumb keep block sizes to pwoers of 2.
	// At this stage there is no need to use fabs again, as all speeds were taken absolutely
	if (HORIZONTAL_K_BLOCKSIZE > 64 )
	{
		if (tid < (HORIZONTAL_K_BLOCKSIZE+1)/2)
			waveSpeedsX[tid] = fmax(waveSpeedsX[tid], waveSpeedsX[tid + HORIZONTAL_K_BLOCKSIZE/2]);
		__syncthreads();
	}
	if (HORIZONTAL_K_BLOCKSIZE/2 > 64 )
	{
		if (tid < (HORIZONTAL_K_BLOCKSIZE+3)/4)
			waveSpeedsX[tid] = fmax(waveSpeedsX[tid], waveSpeedsX[tid + HORIZONTAL_K_BLOCKSIZE/4]);
		__syncthreads();
	}
	if (HORIZONTAL_K_BLOCKSIZE/4 > 64 )
	{
		if (tid < (HORIZONTAL_K_BLOCKSIZE+7)/8)
			waveSpeedsX[tid] = fmax(waveSpeedsX[tid], waveSpeedsX[tid + HORIZONTAL_K_BLOCKSIZE/8]);
		__syncthreads();
	}
	if (HORIZONTAL_K_BLOCKSIZE/8 > 64 )
	{
		if (tid < (HORIZONTAL_K_BLOCKSIZE+15)/16)
			waveSpeedsX[tid] = fmax(waveSpeedsX[tid], waveSpeedsX[tid + HORIZONTAL_K_BLOCKSIZE/16]);
		__syncthreads();
	}
	if (tid < 32)
		warpReduce<blockSize>(waveSpeedsX, tid);

	if (grid_valid && threadIdx.x > 1)
	{
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			param.setElement_q(row, col, k, param.getElement_qNew(row, col, k) - dt/param.dx * (amdq[k] + getSharedWave(wavesX, threadIdx.y, threadIdx.x-1, 0, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX)));
		}
	}

	if (tid == 0)
		param.waveSpeedsX[blockIdx.x + blockIdx.y*gridDim.x] =  waveSpeedsX[0];
}

template <const int numStates, const int numWaves, const int numCoeff, const unsigned int blockSize, class Riemann, class Limiter, class Ent>
__global__ void Riemann_vertical_kernel(pdeParam param, Riemann Riemann_solver_v, Limiter phi, Ent entropy_fixer)
{
	// Every threads gets a unique interface
	// a thread at (row, col) treats the interface between cells
	// (row+1,col)
	// -----------  << thread (row,col)
	// (row  ,col)
	int col = threadIdx.x + blockIdx.x*blockDim.x;
	int row = threadIdx.y + blockIdx.y*blockDim.y - 3*blockIdx.y;

	real dt = *param.dt;

	real apdq[numStates];
	real amdq[numStates];

	real* wavesY		= &shared_elem[0];
	real* waveSpeedsY	= &shared_elem[blockSize*numWaves*numStates];

	real upCell[numStates];		real upCoeff[numCoeff];
	real downCell[numStates];	real downCoeff[numCoeff];

	if (blockIdx.x > gridDim.x-2 || blockIdx.y > gridDim.y-3)
	{
		#pragma unroll
		for (int i = 0; i < numWaves; i++)
			setSharedSpeed(waveSpeedsY, threadIdx.y, threadIdx.x, i, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX, (real)0.0f);
	}

	bool grid_valid = row < param.cellsY-1 && col < param.cellsX;

	// Riemann solver
	if (grid_valid)	// if there are 512 cells in Y 511 threads are in use
	{
		#pragma unroll
		for (int i = 0; i < numStates; i++)
		{
			//downCell[i] = param.getElement_qNew(row,col,i);	// use original data
			//upCell[i] = param.getElement_qNew(row+1,col,i);
			downCell[i] = param.getElement_q(row,col,i);	// use intermediate data
			upCell[i] = param.getElement_q(row+1,col,i);
		}
		#pragma unroll
		for (int i = 0; i < numCoeff; i++)
		{
			downCoeff[i] = param.getElement_coeff(row,col,i);
			upCoeff[i] = param.getElement_coeff(row+1,col,i);
		}
		Riemann_solver_v(	downCell,			// input comes from global memory
							upCell,				//
							downCoeff,			//
							upCoeff,			//
							threadIdx.y,		//
							threadIdx.x,		//
							numStates,			//
							numWaves,			//
							wavesY,				// output to shared memory
							waveSpeedsY);		//

	}

	grid_valid = ( 0 < threadIdx.y && threadIdx.y < VERTICAL_K_BLOCKSIZEY-1 ) && grid_valid;

	if (grid_valid)
	{
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			amdq[k] = 0;
			apdq[k] = 0;
		}
		if (param.entropy_fix)
		{
			//entropy_fix_Shallow_Water_vertical my_entropy_fixer;
			//entropy_fixing_v<numStates, numWaves, entropy_fix_Shallow_Water_vertical>(my_entropy_fixer, wavesY, waveSpeedsY, downCell, upCell, downCoeff, upCoeff, amdq, apdq, threadIdx.y, threadIdx.x);

			entropy_fixing_v<numStates, numWaves>(entropy_fixer, wavesY, waveSpeedsY, downCell, upCell, downCoeff, upCoeff, amdq, apdq, threadIdx.y, threadIdx.x);

			// Straight out for Burger's equation's entropy fix
			/*real waveSpeed = getSharedSpeed(waveSpeedsY, threadIdx.y, threadIdx.x, 0, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);;

			if(waveSpeed < 0.0f)
				amdq[0] = waveSpeed * getSharedWave(wavesY, threadIdx.y, threadIdx.x, 0, 0, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
			else
				apdq[0] = waveSpeed * getSharedWave(wavesY, threadIdx.y, threadIdx.x, 0, 0, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);

			if(upCell[0] < 0 && downCell[0] > 0)
			{
				real theta = 0.0f;
				amdq[0] = -0.5f*sin(downCoeff[0])*downCell[0]*downCell[0];
				apdq[0] =  0.5f*sin(downCoeff[0])*upCell[0]*upCell[0];
			}*/
			if (param.second_order)
			{
				second_order_update_vertical<numStates, numWaves, VERTICAL_K_BLOCKSIZEX, VERTICAL_K_BLOCKSIZEY>(param, wavesY, waveSpeedsY, amdq, apdq, threadIdx.y, threadIdx.x, dt, phi);
				/*/#pragma unroll
				for (int w = 0; w < numWaves; w++)
				{
					real waveSpeed =  getSharedSpeed(waveSpeedsY, threadIdx.y, threadIdx.x, w, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);

					if (waveSpeed < (real)0.0f)
					{
						real limiting_factor = limiting_shared_v_d<numStates, numWaves, VERTICAL_K_BLOCKSIZEX, VERTICAL_K_BLOCKSIZEY>(phi, wavesY, threadIdx.y, threadIdx.x, w);
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesY, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
							amdq[k] +=	//waveSpeed * wave_state
									+
										-(real)0.5f*waveSpeed*(1+(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
						}
					}
					else
					{
						real limiting_factor = limiting_shared_v_u<numStates, numWaves, VERTICAL_K_BLOCKSIZEX, VERTICAL_K_BLOCKSIZEY>(phi, wavesY, threadIdx.y, threadIdx.x, w);
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesY, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
							apdq[k] +=	//waveSpeed * wave_state
									-
										(real)0.5f*waveSpeed*(1-(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
						}
					}
				}/**/
			}
		}
		else
		{
			if (!param.second_order)
			{
				first_order_update<numStates, numWaves, VERTICAL_K_BLOCKSIZEX, VERTICAL_K_BLOCKSIZEY>(wavesY, waveSpeedsY, amdq, apdq, threadIdx.y, threadIdx.x);
				/*/#pragma unroll
				for (int w = 0; w < numWaves; w++)
				{
					real waveSpeed =  getSharedSpeed(waveSpeedsY, threadIdx.y, threadIdx.x, w, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
					if (waveSpeed < (real)0.0f)
					{
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesY, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
							amdq[k] +=	waveSpeed * wave_state;
						}
					}
					else
					{
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesY, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
							apdq[k] +=	waveSpeed * wave_state;
						}
					}
				}/**/
			}
			else	// simple first order scheme
			{
				first_second_order_update_vertical<numStates, numWaves, VERTICAL_K_BLOCKSIZEX, VERTICAL_K_BLOCKSIZEY>(param, wavesY, waveSpeedsY, amdq, apdq, threadIdx.y, threadIdx.x, dt, phi);
				/*/#pragma unroll
				for (int w = 0; w < numWaves; w++)
				{
					real waveSpeed =  getSharedSpeed(waveSpeedsY, threadIdx.y, threadIdx.x, w, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);

					if (waveSpeed < (real)0.0f)
					{
						real limiting_factor = limiting_shared_v_d<numStates, numWaves, VERTICAL_K_BLOCKSIZEX, VERTICAL_K_BLOCKSIZEY>(phi, wavesY, threadIdx.y, threadIdx.x, w);
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesY, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
							amdq[k] +=	waveSpeed * wave_state
									+
										-(real)0.5f*waveSpeed*(1+(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
						}
					}
					else
					{
						real limiting_factor = limiting_shared_v_u<numStates, numWaves, VERTICAL_K_BLOCKSIZEX, VERTICAL_K_BLOCKSIZEY>(phi, wavesY, threadIdx.y, threadIdx.x, w);
						#pragma unroll
						for (int k = 0; k < numStates; k++)
						{
							real wave_state = getSharedWave(wavesY, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
							apdq[k] +=	waveSpeed * wave_state
									-
										(real)0.5f*waveSpeed*(1-(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
						}
					}
				}/**/
			}
		}
	}
	__syncthreads();

	// write the apdq to shared memory
	#pragma unroll
	for(int k = 0; k < numStates; k++)
	{
		setSharedWave(wavesY, threadIdx.y, threadIdx.x, 0, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX, apdq[k]);
	}

	// Local Reduce over Wave Speeds
	// Stage 1
	// Bringing down the number of elements to compare to block size
	int tid = threadIdx.x + threadIdx.y*VERTICAL_K_BLOCKSIZEX;
	// See horizontal version for comments
	waveSpeedsY[tid] = fabs(waveSpeedsY[tid]);
	#pragma unroll
	for (int i = 1; i < numWaves; i++)
	{
		// blockDim.x * blockDim.y is usally a multiple of 32, so there should be no bank conflicts.
		waveSpeedsY[tid] = fmax(waveSpeedsY[tid], fabs(waveSpeedsY[tid + i*blockSize]));
	}
	__syncthreads();

	if (VERTICAL_K_BLOCKSIZE > 64)
	{
		if (tid < (VERTICAL_K_BLOCKSIZE+1)/2)
			waveSpeedsY[tid] = fmax(waveSpeedsY[tid], waveSpeedsY[tid + VERTICAL_K_BLOCKSIZE/2]);
		__syncthreads();
	}
	if (VERTICAL_K_BLOCKSIZE/2 > 64)
	{
		if (tid < (VERTICAL_K_BLOCKSIZE+3)/4)
			waveSpeedsY[tid] = fmax(waveSpeedsY[tid], waveSpeedsY[tid + VERTICAL_K_BLOCKSIZE/4]);
		__syncthreads();
	}
	if (VERTICAL_K_BLOCKSIZE/4 > 64)
	{
		if (tid < (VERTICAL_K_BLOCKSIZE+7)/8)
			waveSpeedsY[tid] = fmax(waveSpeedsY[tid], waveSpeedsY[tid + VERTICAL_K_BLOCKSIZE/8]);
		__syncthreads();
	}
	if (VERTICAL_K_BLOCKSIZE/8 > 64)
	{
		if (tid < (VERTICAL_K_BLOCKSIZE+15)/16)
			waveSpeedsY[tid] = fmax(waveSpeedsY[tid], waveSpeedsY[tid + VERTICAL_K_BLOCKSIZE/16]);
		__syncthreads();
	}
	if (tid < 32)
		warpReduce<blockSize>(waveSpeedsY, tid);


	if (grid_valid && threadIdx.y > 1)
	{
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			param.setElement_q(row, col, k, param.getElement_q(row, col, k) - dt/param.dy * (amdq[k] + getSharedWave(wavesY, threadIdx.y-1, threadIdx.x, 0, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX)));
		}
	}

	if (tid == 0)
		param.waveSpeedsY[blockIdx.x + blockIdx.y*gridDim.x] =  waveSpeedsY[0];
}

__global__ void timeStepAdjust_simple(pdeParam param)
{
	// check CFL violation and determine next time step dt
	real u = param.waveSpeedsX[0];
	real v = param.waveSpeedsY[0];
	real dx = param.dx;
	real dy = param.dy;
	real dt = *param.dt;
	real dt_used = *param.dt_used;
	if (dt*fmax(u/dx,v/dy) > 1.0f)	// if CFL condition violated
	{
		// simulation did not advance, no time is to be incremented to the global simulation time
		dt_used = 0.0f;
		// compute a new dt with a stricter assumption
		dt = param.CFL_lower_bound/fmax(u/dx,v/dy);

		// condition failed do not swap buffers, effectively reverts the time step
		*param.revert = false;
	}
	else	// else if no violation was recorded
	{
		// remember the time step used to increment the global simulation time
		dt_used = dt;
		// compute the the next dt to be used
		dt = param.desired_CFL/fmax(u/dx,v/dy);

		// allow buffer swap
		*param.revert = true;
	}
	*param.dt = /*0.0001f;/*/dt/**/;
	*param.dt_used = /*0.0001f;/*/dt_used/***/;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////   Wrapper Function   ////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Riemann_h, class Riemann_v, class Limiter, class Ent_h, class Ent_v>
int limited_Riemann_Update(pdeParam &param,						// Problem parameters
			   Riemann_h Riemann_pointwise_solver_h,	//
			   Riemann_v Riemann_pointwise_solver_v,	//
			   Limiter limiter_phi,						//
			   Ent_h entropy_fix_h,
			   Ent_v entropy_fix_v
			   )
{
    {
	// RIEMANN, FLUCTUATIONS and UPDATES
	const unsigned int blockDim_XR = HORIZONTAL_K_BLOCKSIZEX;
	const unsigned int blockDim_YR = HORIZONTAL_K_BLOCKSIZEY;
	unsigned int gridDim_XR = (param.cellsX + (blockDim_XR-3-1)) / (blockDim_XR-3);
	unsigned int gridDim_YR = (param.cellsY + (blockDim_YR-1)) / (blockDim_YR);
	dim3 dimGrid_hR(gridDim_XR, gridDim_YR);
	dim3 dimBlock_hR(blockDim_XR, blockDim_YR);
	int shared_mem_size = HORIZONTAL_K_BLOCKSIZEX*HORIZONTAL_K_BLOCKSIZEY*NUMWAVES*(NUMSTATES+1)*sizeof(real);
	
	Riemann_horizontal_kernel<NUMSTATES, NUMWAVES, NUMCOEFF, HORIZONTAL_K_BLOCKSIZEX*HORIZONTAL_K_BLOCKSIZEY, Riemann_h, Limiter><<<dimGrid_hR, dimBlock_hR, shared_mem_size>>>(param, Riemann_pointwise_solver_h, limiter_phi, entropy_fix_h);
	CHKERR();
	
	// REDUCTION
	const unsigned int blockDim_X = 512;		// fine tune the best block size
	
	size_t SharedMemorySize = (blockDim_X)*sizeof(real);
	unsigned int gridDim_X1;
	
	gridDim_X1 = 1;
	
	dim3 dimGrid1(gridDim_X1);
	dim3 dimBlock1(blockDim_X);
	
	reduceMax_simplified<blockDim_X><<< dimGrid1, dimBlock1, SharedMemorySize>>>(param.waveSpeedsX, gridDim_XR*gridDim_YR);
    }
    {
	// RIEMANN, FLUCTUATIONS and UPDATE
	const unsigned int blockDim_XR = VERTICAL_K_BLOCKSIZEX;
	const unsigned int blockDim_YR = VERTICAL_K_BLOCKSIZEY;
	unsigned int gridDim_XR = (param.cellsX + (blockDim_XR-1)) / (blockDim_XR);
	unsigned int gridDim_YR = (param.cellsY + (blockDim_YR-3-1)) / (blockDim_YR-3);
	dim3 dimGrid_vR(gridDim_XR, gridDim_YR);
	dim3 dimBlock_vR(blockDim_XR, blockDim_YR);
	int shared_mem_size = VERTICAL_K_BLOCKSIZEX*VERTICAL_K_BLOCKSIZEY*NUMWAVES*(NUMSTATES+1)*sizeof(real);
	
	Riemann_vertical_kernel<NUMSTATES, NUMWAVES, NUMCOEFF, VERTICAL_K_BLOCKSIZEX*VERTICAL_K_BLOCKSIZEY, Riemann_v, Limiter><<<dimGrid_vR, dimBlock_vR,shared_mem_size>>>(param, Riemann_pointwise_solver_v, limiter_phi,entropy_fix_v);
	
	// REDUCTION
	const unsigned int blockDim_X = 512;		// fine tune the best block size
	
	size_t SharedMemorySize = (blockDim_X)*sizeof(real);
	unsigned int gridDim_X2;
	
	gridDim_X2 = 1;
	
	dim3 dimGrid2(gridDim_X2);
	dim3 dimBlock2(blockDim_X);
	
	reduceMax_simplified<blockDim_X><<< dimGrid2, dimBlock2, SharedMemorySize>>>(param.waveSpeedsY, gridDim_XR*gridDim_YR);
    }
    {
	timeStepAdjust_simple<<<1,1>>>(param);
	
	bool revert;
	cudaMemcpy(&revert, param.revert, sizeof(bool), cudaMemcpyDeviceToHost);
	if (revert)
	    {
		// Swap q and qNew before stepping again
		// At this stage qNew became old and q has the latest state that is
		// because q was updated based on qNew, which right before 'step'
		// held the latest update.
		real* temp = param.qNew;
		param.qNew = param.q;
		param.q = temp;
	    }
    }
    
    return 0;
}

#endif

			//// Comments on this part
			//// ---------------------
			//// With entropy fix, the computation of first order fluctuations (amdq and apdq) cannot be integrated with the second order correctionas elegantly.
			//// This is because computing the first order fluctuations requires more than checking the wave speed sign and the way we want to implement its
			//// computation requires it to be computed completely before moving on to second order corrections which is in fact quite independent from the first
			//// order fluctuations, in terms of data dependence (except if one wants to try to use the special wave speeds that come from the entropy fix in the
			//// correction process).
			//// We compute first and second order fluctuations as a single process, which in fact can be split using this test. This testing code simply computes
			//// first order fluctuations and then moves on for second order corrections, and would be replaced by an entropy fixing method when necessary, without
			//// interfering with second order computations, making this a more elegant solution to the problem (instead of splitting the method into 2 large chunks
			//// of code). However, this involves redundant reads, which reflect according to a few test to around 1-2% in decrease in performance. It remains to see
			//// if this loss is compensated by the cleaness of the approach once entropy fix is required.
			////
			//// Same for vertical counterpart
			//#pragma unroll
			//for (int w = 0; w < numWaves; w++)
			//{
			//	real waveSpeed =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

			//	if (waveSpeed < (real)0.0f)
			//	{
			//		#pragma unroll
			//		for (int k = 0; k < numStates; k++)
			//		{
			//			real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//			amdq[k] +=	waveSpeed * wave_state;
			//		}
			//	}
			//	else
			//	{
			//		#pragma unroll
			//		for (int k = 0; k < numStates; k++)
			//		{
			//			real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//			apdq[k] +=	waveSpeed * wave_state;
			//		}
			//	}
			//}

			/////////////////////////////// Entropy Fix Check START
			//bool Burger_equation  = false;
			//bool entropy_suspicious  = false;
			//for (int w = 0; w < numWaves && !entropy_suspicious; w++)
			//{
			//	real waveSpeed_left  =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x-1, 0, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//	real waveSpeed_right =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x+1, 0, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

			//	entropy_suspicious = waveSpeed_left < 0 && 0 < waveSpeed_right;

			//	//real waveSpeed =  getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_BLOCKSIZEX);

			//	//// check opposite direction of the wavespeed
			//	//if ( waveSpeed < 0 )
			//	//{
			//	//	// check with right
			//	//	real waveSpeed_toRight = getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x+1, w, numWaves, HORIZONTAL_BLOCKSIZEX);
			//	//	if ( waveSpeed_toRight > 0 )
			//	//		entropy_suspicious = true;
			//	//}
			//	//else
			//	//{
			//	//	// check with left
			//	//	real waveSpeed_toLeft = getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x-1, w, numWaves, HORIZONTAL_BLOCKSIZEX);
			//	//	if ( waveSpeed_toLeft < 0 )
			//	//		entropy_suspicious = true;
			//	//}
			//}
			///////////////////////////////////// Entropy Fix Check END
			///*__shared__ int warp_ids[6];
			//warp_ids [0] = 0;
			//warp_ids [1] = 0;
			//warp_ids [2] = 0;
			//warp_ids [3] = 0;
			//warp_ids [4] = 0;
			//warp_ids [5] = 0;*/
			//if (param.entropy_fix && entropy_suspicious && !Burger_equation)
			//{
			//	//int warp_index = (threadIdx.x + threadIdx.y*blockDim.x)%32;
			//	//if ( atomicCAS(&warp_ids[warp_index], 0, 1) == 0 )
			//	//{
			//	//	atomicAdd(entropy_fix_entering_warps, 1);
			//	//}
			//	//atomicAdd(entropy_fix_entering_threads, 1);
			//	bool entropy_fixed = false;

			//	///////////////////////////////////// Entropy Fix START
			//	real lambda_l = leftCell[1]/leftCell[0] - sqrt(9.8f*leftCell[0]);
			//	real lambda_r;
			//	real s_fraction;
			//	if (lambda_l > 0 && getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, 0, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX) > 0.0f)
			//		entropy_fixed = true;

			//	int w = 0;	// wave number
			//	while (!entropy_fixed && w < numWaves)
			//	{
			//		// solve for the wave speed just to the left and just to the right of the current wave_w
			//		#pragma unroll
			//		for (int k = 0; k < numStates; k++)
			//		{
			//			real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//			leftCell[k] += wave_state;
			//		}

			//		// compute lambda_r according to the eigenvalues of the jacobian of the differential operator at the new state q right after the w-th wave
			//		if ( w == 0 )
			//			lambda_r = leftCell[1]/leftCell[0] - sqrt(9.8f*leftCell[0]);
			//		else if (w == 1 )
			//			lambda_r = getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//		else
			//			lambda_r = leftCell[1]/leftCell[0] + sqrt(9.8f*leftCell[0]);


			//		real waveSpeed =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//		if ( lambda_l < 0 && 0 < lambda_r )
			//		{
			//			s_fraction = lambda_l*(lambda_r - waveSpeed)/(lambda_r - lambda_l);
			//			entropy_fixed = true;
			//			//atomicAdd(entropy_fix_entering_threads, -1);
			//		}
			//		else if ( waveSpeed < 0 )
			//		{
			//			s_fraction = waveSpeed;
			//		}
			//		else
			//		{
			//			s_fraction = 0.0f;
			//		}

			//		#pragma unroll
			//		for (int k = 0; k < numStates; k++)
			//		{
			//			real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//			amdq[k] += s_fraction * wave_state;
			//		}

			//		lambda_l = lambda_r;
			//		w++;
			//	}

			//	// Sum all the waves into apdq, then take out computed amdq from it
			//	#pragma unroll
			//	for(int w = 0; w < numWaves; w++)
			//	{
			//		real waveSpeed =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

			//		#pragma unroll
			//		for(int k = 0; k < numStates; k++)
			//		{
			//			real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//			apdq[k] += waveSpeed * wave_state;
			//		}
			//	}
			//	#pragma unroll
			//	for(int k = 0; k < numStates; k++)
			//	{
			//		apdq[k] -= amdq[k];
			//	}
			//	// /////////////////////////////////// Entropy Fix END*/
			//}
			//else if (Burger_equation)
			//{
			//	//bool entropy_fixed = false;

			//	//real lambda_l = cos(0.525f)*leftCell[0];
			//	//real lambda_r;
			//	//real s_fraction;
			//	//if (lambda_l > 0 && getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, 0, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX) > 0.0f)
			//	//	entropy_fixed = true;

			//	//int w = 0;	// wave number
			//	//while (!entropy_fixed && w < numWaves)
			//	//{
			//	//	// solve for the wave speed just to the left and just to the right of the current wave_w
			//	//	#pragma unroll
			//	//	for (int k = 0; k < numStates; k++)
			//	//	{
			//	//		real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//	//		leftCell[k] += wave_state;
			//	//	}

			//	//	// compute lambda_r according to the eigenvalues of the jacobian of the differential operator at the new state q right after the w-th wave
			//	//	if ( w == 0 )
			//	//		lambda_r = cos(0.525f)*leftCell[0];


			//	//	real waveSpeed =  getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//	//	if ( lambda_l < 0 && 0 < lambda_r )
			//	//	{
			//	//		s_fraction = lambda_l*(lambda_r - waveSpeed)/(lambda_r - lambda_l);
			//	//		entropy_fixed = true;
			//	//	}
			//	//	else if ( waveSpeed < 0 )
			//	//	{
			//	//		s_fraction = waveSpeed;
			//	//	}
			//	//	else
			//	//	{
			//	//		s_fraction = 0.0f;
			//	//	}

			//	//	#pragma unroll
			//	//	for (int k = 0; k < numStates; k++)
			//	//	{
			//	//		real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//	//		amdq[k] += s_fraction * wave_state;
			//	//	}

			//	//	lambda_l = lambda_r;
			//	//	w++;
			//	//}

			//	//// Sum all the waves into apdq, then take out computed amdq from it
			//	//#pragma unroll
			//	//for(int w = 0; w < numWaves; w++)
			//	//{
			//	//	real waveSpeed = getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

			//	//	#pragma unroll
			//	//	for(int k = 0; k < numStates; k++)
			//	//	{
			//	//		real wave_state = getSharedWave(wavesX, threadIdx.y, threadIdx.x, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			//	//		apdq[k] += waveSpeed * wave_state;
			//	//	}
			//	//}
			//	//#pragma unroll
			//	//for(int k = 0; k < numStates; k++)
			//	//{
			//	//	apdq[k] -= amdq[k];
			//	//}
			//}

			/*real waveSpeed = getSharedSpeed(waveSpeedsX, threadIdx.y, threadIdx.x, 0, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);;

			if(waveSpeed < 0.0f)
				amdq[0] = waveSpeed * getSharedWave(wavesX, threadIdx.y, threadIdx.x, 0, 0, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
			else
				apdq[0] = waveSpeed * getSharedWave(wavesX, threadIdx.y, threadIdx.x, 0, 0, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

			if(leftCell[0] < 0 && rightCell[0] > 0)
			{
				real theta = 0.0f;
				amdq[0] = -0.5f*cos(leftCoeff[0])*leftCell[0]*leftCell[0];
				apdq[0] =  0.5f*cos(leftCoeff[0])*rightCell[0]*rightCell[0];
			}*/
