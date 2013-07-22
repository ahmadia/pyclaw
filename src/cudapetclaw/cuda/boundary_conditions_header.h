#ifndef __BOUNDARY_CONDITIONS_HEADER_H__
#define __BOUNDARY_CONDITIONS_HEADER_H__

#include "common.h"

template<class L, class R, class U, class D>
    class boundaryConditions
{
 public:
    L condition_left;
    R condition_right;
    U condition_up;
    D condition_down;
};


struct BC_none
{
  __device__ void operator()(pdeParam param, int ignored)
  {
    return;
  }
};

//////////////////////////////////////// ABSORBING CONDITIONS //////////////////////////////////////////////////////////

// The idea behind these boundry condition kernels is to have a set of threads work on the edges
// For example for the left edge,  threads will be created equal to the amount of rows there are
// each will handle the boundary cells on its assigned row.
// ! The number of rows (or columns) includes the boundaries on both sides
struct BC_left_absorbing
{
    __device__ void operator()(pdeParam param, int row)
    {
	int range = param.cellsY;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;

	if ( row < range )							// the number of working threads must be equal to the number of existing rows
	    for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
		for ( int k = 0; k < param.numStates; k++)			// could be modified to be handled by threads in the z directions
		    param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_end, k));
    }
};
struct BC_right_absorbing
{
    __device__ void operator()(pdeParam param, int row)
    {
	int range = param.cellsY;
	int boundary_start = param.cellsX - param.ghostCells;
	int boundary_end = param.cellsX;

	if ( row < range )							// the number of working threads must be equal to the number of existing rows
	    for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
		for ( int k = 0; k < param.numStates; k++)			// could be modified to be handled by threads in the z directions
		    param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_start-1, k));
    }
};
struct BC_up_absorbing
{
    __device__ void operator()(pdeParam param, int col)
    {
	int range = param.cellsX;
	int boundary_start = param.cellsY - param.ghostCells;
	int boundary_end = param.cellsY;

	if ( col < range )							// the number of working threads must be equal to the number of existing rows
	    for (int row = boundary_start; row < boundary_end; row++)		// could be modified to be handled by threads in the y directions
		for ( int k = 0; k < param.numStates; k++)		        // could be modified to be handled by threads in the z directions
		    param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_start-1, col, k));
    }
};
struct BC_down_absorbing
{
    __device__ void operator()(pdeParam param, int col)
    {
	int range = param.cellsX;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;

	if ( col < range )							// the number of working threads must be equal to the number of existing rows
	    for (int row = boundary_start; row < boundary_end; row++)		// could be modified to be handled by threads in the y directions
		for ( int k = 0; k < param.numStates; k++)		        // could be modified to be handled by threads in the z directions
		    param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_end, col, k));
    }
};

//////////////////////////////////////// REFLECTIVE CONDITIONS //////////////////////////////////////////////////////////

struct BC_left_reflective
{
    __device__ void operator()(pdeParam param, int row)
    {
	int range = param.cellsY;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;
	int boundary_length = boundary_end - boundary_start;

	if ( row < range )
	    for (int col = boundary_start; col < boundary_end; col++)
		{
		    param.setElement_qNew(row, col, 0, param.getElement_qNew(row, boundary_end + boundary_length - 1 - col, 0));
		    for ( int k = 1; k < param.numStates; k++)
			param.setElement_qNew(row, col, k,-param.getElement_qNew(row, boundary_end + boundary_length - 1 - col, k));
		}
    }
};
struct BC_right_reflective
{
    __device__ void operator()(pdeParam param, int row)
    {
	int range = param.cellsY;
	int boundary_start = param.cellsX - param.ghostCells;
	int boundary_end = param.cellsX;
	int reverse_counter = 1;

	if ( row < range )
	    for (int col = boundary_start; col < boundary_end; col++, reverse_counter++)
		{
		    param.setElement_qNew(row, col, 0, param.getElement_qNew(row, boundary_start - reverse_counter, 0));

		    for ( int k = 1; k < param.numStates; k++)
			param.setElement_qNew(row, col, k,-param.getElement_qNew(row, boundary_start - reverse_counter, k));
		}
    }
};
struct BC_up_reflective
{
    __device__ void operator()(pdeParam param, int col)
    {
	int range = param.cellsX;
	int boundary_start = param.cellsY - param.ghostCells;
	int boundary_end = param.cellsY;
	int reverse_counter = 1;


	if ( col < range )
	    for (int row = boundary_start; row < boundary_end; row++, reverse_counter++)
		{
		    param.setElement_qNew(row, col, 0, param.getElement_qNew(boundary_start - reverse_counter, col, 0));
		    for (int k = 1; k < param.numStates; k++)
			param.setElement_qNew(row, col, k,-param.getElement_qNew(boundary_start - reverse_counter, col, k));
		}
    }
};
struct BC_down_reflective
{
    __device__ void operator()(pdeParam param, int col)
    {
	int range = param.cellsX;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;
	int reverse_counter = param.ghostCells - 1;

	if ( col < range )
	    for (int row = boundary_start; row < boundary_end; row++, reverse_counter--)
		{
		    param.setElement_qNew(row, col, 0, param.getElement_qNew(boundary_end + reverse_counter, col, 0));
		    for ( int k = 1; k < param.numStates; k++)
			param.setElement_qNew(row, col, k,-param.getElement_qNew(boundary_end + reverse_counter, col, k));
		}
    }
};

#endif
