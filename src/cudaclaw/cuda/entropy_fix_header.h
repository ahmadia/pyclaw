#ifndef __ENTROPY_FIX_HEADERS_H__
#define __ENTROPY_FIX_HEADERS_H__

#include "auxiliary_device_functions.h"

template<const int numStates, const int numWaves, class Ent>
    __device__ real entropy_fixing_h(Ent entropy_fix, real* waves, real* waveSpeeds, real* q_left, real* q_right, real* u_left, real* u_right, real* amdq, real* apdq, int row, int col)
{
    bool entropy_fixed = false;
    ///////////////////////////////////// Entropy Fix START
    real lambda_l = entropy_fix(waveSpeeds, q_left, q_right, u_left, u_right, 0, row, col, numWaves);
    real lambda_r;
    real s_fraction;
    if (lambda_l > 0 && getSharedSpeed(waveSpeeds, row, col, 0, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX) > 0.0f)
	entropy_fixed = true;

    int w = 0;	// wave number
    while (!entropy_fixed && w < numWaves)
	{
	    // solve for the wave speed just to the left and just to the right of the current wave_w
#pragma unroll
	    for (int k = 0; k < numStates; k++)
		{
		    real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
		    q_left[k] += wave_state;
		}

	    // compute lambda_r according to the eigenvalues of the jacobian of the differential operator at the new state q right after the w-th wave
	    lambda_r = entropy_fix(waveSpeeds, q_left, q_right, u_left, u_right, w, row, col, numWaves);

	    real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
	    if ( lambda_l < 0 && 0 < lambda_r )
		{
		    s_fraction = lambda_l*(lambda_r - waveSpeed)/(lambda_r - lambda_l);
		    entropy_fixed = true;
		}
	    else if ( waveSpeed < 0 )
		{
		    s_fraction = waveSpeed;
		}
	    else
		{
		    s_fraction = 0.0f;
		}

#pragma unroll
	    for (int k = 0; k < numStates; k++)
		{
		    real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
		    amdq[k] += s_fraction * wave_state;
		}

	    lambda_l = lambda_r;
	    w++;
	}

    // Sum all the waves into apdq, then take out computed amdq from it
#pragma unroll
    for(int w = 0; w < numWaves; w++)
	{
	    real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);

#pragma unroll
	    for(int k = 0; k < numStates; k++)
		{
		    real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
		    apdq[k] += waveSpeed * wave_state;
		}
	}
#pragma unroll
    for(int k = 0; k < numStates; k++)
	{
	    apdq[k] -= amdq[k];
	}
    // /////////////////////////////////// Entropy Fix END*/
}
template<const int numStates, const int numWaves, class Ent>
    __device__ real entropy_fixing_v(Ent entropy_fix, real* waves, real* waveSpeeds, real* q_left, real* q_right, real* u_left, real* u_right, real* amdq, real* apdq, int row, int col)
{

    bool entropy_fixed = false;
    ///////////////////////////////////// Entropy Fix START
    real lambda_l = entropy_fix(waveSpeeds, q_left, q_right, u_left, u_right, 0, row, col, numWaves);
    real lambda_r;
    real s_fraction;
    if (lambda_l > 0 && getSharedSpeed(waveSpeeds, row, col, 0, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX) > 0.0f)
	entropy_fixed = true;

    int w = 0;	// wave number
    while (!entropy_fixed && w < numWaves)
	{
	    // solve for the wave speed just to the left and just to the right of the current wave_w
#pragma unroll
	    for (int k = 0; k < numStates; k++)
		{
		    real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
		    q_left[k] += wave_state;
		}

	    // compute lambda_r according to the eigenvalues of the jacobian of the differential operator at the new state q right after the w-th wave
	    lambda_r = entropy_fix(waveSpeeds, q_left, q_right, u_left, u_right, w, row, col, numWaves);

	    real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
	    if ( lambda_l < 0 && 0 < lambda_r )
		{
		    s_fraction = lambda_l*(lambda_r - waveSpeed)/(lambda_r - lambda_l);
		    entropy_fixed = true;
		}
	    else if ( waveSpeed < 0 )
		{
		    s_fraction = waveSpeed;
		}
	    else
		{
		    s_fraction = 0.0f;
		}

#pragma unroll
	    for (int k = 0; k < numStates; k++)
		{
		    real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
		    amdq[k] += s_fraction * wave_state;
		}

	    lambda_l = lambda_r;
	    w++;
	}

    // Sum all the waves into apdq, then take out computed amdq from it
#pragma unroll
    for(int w = 0; w < numWaves; w++)
	{
	    real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);

#pragma unroll
	    for(int k = 0; k < numStates; k++)
		{
		    real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
		    apdq[k] += waveSpeed * wave_state;
		}
	}
#pragma unroll
    for(int k = 0; k < numStates; k++)
	{
	    apdq[k] -= amdq[k];
	}
    // /////////////////////////////////// Entropy Fix END*/
}
struct entropy_fix_Shallow_Water_horziontal
{
    __device__ real operator() (real* waveSpeeds, real* q_left, real* q_right, real* u_left, real* u_right, int lambda, int row, int col, int numWaves)
    {
	real lambda_r = 0.0f;

	if (lambda == 0)
	    lambda_r = q_left[1]/q_left[0] - sqrt(9.8f*q_left[0]);
	else if (lambda == 1)
	    lambda_r = getSharedSpeed(waveSpeeds, row, col, 1, numWaves, HORIZONTAL_K_BLOCKSIZEY, HORIZONTAL_K_BLOCKSIZEX);
	else if (lambda == 2)
	    lambda_r = q_left[1]/q_left[0] + sqrt(9.8f*q_left[0]);

	return lambda_r;
    }
};
struct entropy_fix_Shallow_Water_vertical
{
    __device__ real operator() (real* waveSpeeds, real* q_left, real* q_right, real* u_left, real* u_right, int lambda, int row, int col, int numWaves)
    {
	real lambda_r = 0.0f;
	if (lambda == 0)
	    lambda_r = q_left[2]/q_left[0] - sqrt(9.8f*q_left[0]);
	else if (lambda == 1)
	    lambda_r = getSharedSpeed(waveSpeeds, row, col, 1, numWaves, VERTICAL_K_BLOCKSIZEY, VERTICAL_K_BLOCKSIZEX);
	else if (lambda == 2)
	    lambda_r = q_left[2]/q_left[0] + sqrt(9.8f*q_left[0]);

	return lambda_r;
    }
};

struct entropy_fix_Burger_horizontal
{
    inline __device__ real operator() (real* waveSpeeds, real* q_left, real* q_right, real* u_left, real* u_right, int lambda, int row, int col, int numWaves)
    {
	real lambda_l = cos(u_left[0])*q_left[0];
	return lambda_l;
    }
};
struct entropy_fix_Burger_vertical
{
    inline __device__ real operator() (real* waveSpeeds, real* q_left, real* q_right, real* u_left, real* u_right, int lambda, int row, int col, int numWaves)
    {
	real lambda_l = sin(u_left[0])*q_left[0];
	return lambda_l;
    }
};
struct null_entropy
{
    inline __device__ real operator() (real* waveSpeeds, real* q_left, real* q_right, real* u_left, real* u_right, int lambda, int row, int col, int numWaves)
    {
	return 0;
    }
};

#endif
