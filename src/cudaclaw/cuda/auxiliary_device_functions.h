#ifndef __AUXILIARY_DEVICE_FUNCTIONS_H__
#define __AUXILIARY_DEVICE_FUNCTIONS_H__

template<const int numStates, const int numWaves, const int blockSizeX, const int blockSizeY>
inline __device__ void first_order_update(real* waves, real* waveSpeeds, real* amdq, real* apdq, int row, int col)
{
	#pragma unroll
	for (int w = 0; w < numWaves; w++)
	{
		real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, blockSizeY, blockSizeX);
		if (waveSpeed < (real)0.0f)
		{
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				amdq[k] +=	waveSpeed * wave_state;
			}
		}
		else
		{
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				apdq[k] +=	waveSpeed * wave_state;
			}
		}
	}
}


template<const int numStates, const int numWaves, const int blockSizeX, const int blockSizeY, class Limiter>
inline __device__ void second_order_update_horizontal(pdeParam &param, real* waves, real* waveSpeeds, real* amdq, real* apdq, int row, int col, real dt, Limiter phi)
{
	#pragma unroll
	for (int w = 0; w < numWaves; w++)
	{
		real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, blockSizeY, blockSizeX);

		if (waveSpeed < (real)0.0f)
		{
			real limiting_factor = limiting_shared_h_l<numStates, numWaves, blockSizeX, blockSizeY, Limiter>(phi, waves, row, col, w);

			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				amdq[k] += - (real)0.5f*waveSpeed*(1+(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
			}
		}
		else
		{
			real limiting_factor = limiting_shared_h_r<numStates, numWaves, blockSizeX, blockSizeY, Limiter>(phi, waves, row, col, w);

			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				apdq[k] += - (real)0.5f*waveSpeed*(1-(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
			}
		}
	}
}
template<const int numStates, const int numWaves, const int blockSizeX, const int blockSizeY, class Limiter>
inline __device__ void second_order_update_vertical(pdeParam &param, real* waves, real* waveSpeeds, real* amdq, real* apdq, int row, int col, real dt, Limiter phi)
{
	#pragma unroll
	for (int w = 0; w < numWaves; w++)
	{
		real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, blockSizeY, blockSizeX);

		if (waveSpeed < (real)0.0f)
		{
			real limiting_factor = limiting_shared_v_d<numStates, numWaves, blockSizeX, blockSizeY>(phi, waves, row, col, w);
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				amdq[k] += -(real)0.5f*waveSpeed*(1+(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
			}
		}
		else
		{
			real limiting_factor = limiting_shared_v_u<numStates, numWaves, blockSizeX, blockSizeY>(phi, waves, row, col, w);
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				apdq[k] += - (real)0.5f*waveSpeed*(1-(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
			}
		}
	}
}

template<const int numStates, const int numWaves, const int blockSizeX, const int blockSizeY, class Limiter>
inline __device__ void first_second_order_update_horizontal(pdeParam &param, real* waves, real* waveSpeeds, real* amdq, real* apdq, int row, int col, real dt, Limiter phi)
{
	#pragma unroll
	for (int w = 0; w < numWaves; w++)
	{
		real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, blockSizeY, blockSizeX);

		if (waveSpeed < (real)0.0f)
		{
			real limiting_factor = limiting_shared_h_l<numStates, numWaves, blockSizeX, blockSizeY>(phi, waves, row, col, w);

			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				amdq[k] +=	waveSpeed * wave_state
						+
							-(real)0.5f*waveSpeed*(1+(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
			}
		}
		else
		{
			real limiting_factor = limiting_shared_h_r<numStates, numWaves, blockSizeX, blockSizeY>(phi, waves, row, col, w);

			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				apdq[k] +=	waveSpeed * wave_state
						-
							(real)0.5f*waveSpeed*(1-(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
			}
		}
	}
}
template<const int numStates, const int numWaves, const int blockSizeX, const int blockSizeY, class Limiter>
inline __device__ void first_second_order_update_vertical(pdeParam &param, real* waves, real* waveSpeeds, real* amdq, real* apdq, int row, int col, real dt, Limiter phi)
{
	#pragma unroll
	for (int w = 0; w < numWaves; w++)
	{
		real waveSpeed =  getSharedSpeed(waveSpeeds, row, col, w, numWaves, blockSizeY, blockSizeX);

		if (waveSpeed < (real)0.0f)
		{
			real limiting_factor = limiting_shared_v_d<numStates, numWaves, blockSizeX, blockSizeY>(phi, waves, row, col, w);
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				amdq[k] +=	waveSpeed * wave_state
						+
							-(real)0.5f*waveSpeed*(1+(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
			}
		}
		else
		{
			real limiting_factor = limiting_shared_v_u<numStates, numWaves, blockSizeX, blockSizeY>(phi, waves, row, col, w);
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				real wave_state = getSharedWave(waves, row, col, w, k, numWaves, numStates, blockSizeY, blockSizeX);
				apdq[k] +=	waveSpeed * wave_state
						-
							(real)0.5f*waveSpeed*(1-(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
			}
		}
	}
}

#endif