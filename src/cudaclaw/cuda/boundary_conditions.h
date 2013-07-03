#ifndef __BOUNDARY_CONDITIONS_H__
#define __BOUNDARY_CONDITIONS_H__

#include "boundary_conditions_header.h"

template<class BCS>
__global__ void boundary_kernel(pdeParam param, BCS conditions)
{
    int thread_global_id = threadIdx.x + blockDim.x*blockIdx.x;
    conditions.condition_left(param, thread_global_id);
    conditions.condition_right(param, thread_global_id);
    conditions.condition_up(param, thread_global_id);
    conditions.condition_down(param, thread_global_id);
}

template<class BCS>
void setBoundaryConditions(pdeParam& param, BCS bcs)
{
    unsigned int blockDimensionX = 256;

    int maxSide = (param.cellsX >= param.cellsY)? param.cellsX:param.cellsY;
    unsigned int gridDimensionX = (maxSide+blockDimensionX-1)/blockDimensionX;

    dim3 dimGrid(gridDimensionX);
    dim3 dimBlock(blockDimensionX);

    boundary_kernel<<<dimGrid, dimBlock>>>(param, bcs);
}

#endif
