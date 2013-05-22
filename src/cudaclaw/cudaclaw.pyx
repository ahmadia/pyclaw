# distutils: language = c++

"""
cudaclaw.pyx

Cythonized bindings to CUDA-enabled hyperbolic solvers for PyClaw

"""

# external CUDA interfaces


cdef extern from "cudaclaw.h" :
    ctypedef double real
    cdef extern int shallow_water_solver_allocate(int, int, int,
                                                  int, int, int,
                                                  real, real,
                                                  real, real,
                                                  real, real)
    cdef extern int shallow_water_solver_setup (int bc_left,
                                                int bc_right,
                                                int bc_up,
                                                int bc_down,
                                                int limiter)

    cdef extern int hyperbolic_solver_2d_step (real dt, real* dt)
    cdef extern int hyperbolic_solver_2d_get_qbc (real* qbc)
    cdef extern int hyperbolic_solver_2d_set_qbc (real* qbc)

import cython
import sys
import clawpack.pyclaw

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

def check_err(err):

    if (not err):
        return
    else:
        caller = sys._getframe(1).f_code.co_name
        raise Exception(
            "Error code %d returned from %s" % (err, caller))

class CUDAState(clawpack.pyclaw.State):
    r"""  See the corresponding PyClaw State documentation."""

    # local members
    synced = False

    def _get_qbc(self, np.ndarray[real, ndim=3, mode="fortran"] qbc):
        cdef int err
        err = hyperbolic_solver_2d_get_qbc(&qbc[0,0,0])
        check_err(err)

    def set_q_from_qbc(self, num_ghost, qbc):
        r"""
        Set the value of q using the array qbc. For CUDASolver, this
        involves retrieving from device memory into qbc, reordering back from
        CUDAClaw order to PyClaw order, then copying the appropriate data into q.

        (fastest-moving index first)
        CUDAClaw order: (x, y, e)
        PyClaw   order: (e, x, y)
        """

        cdef int meqn, mx, my

        meqn = qbc.shape[0]
        mx   = qbc.shape[1]
        my   = qbc.shape[2]

        # need to lift this allocation out to a state buffer
        qbc = np.empty((mx,my,meqn), order='F')

        self._get_qbc(qbc)

        qbct = np.rollaxis(qbc, 2, 0)
        qbct = np.asfortranarray(np.reshape(qbct, (meqn, mx, my), order='F'))

        self.q = qbct[:,num_ghost:-num_ghost,num_ghost:-num_ghost]

        self.synced = True


    def _set_qbc(self, np.ndarray[real, ndim=3, mode="fortran"] qbc):

        cdef int err

        err = hyperbolic_solver_2d_set_qbc(&qbc[0,0,0])
        check_err(err)

    def get_qbc_from_q(self, num_ghost, qbc):
        """
        Fills in the interior of qbc by copying q to it.  For CUDASolver,
        this involves copying the appropriate data into qbc, reordering from
        PyClaw order to CUDACLAW order, then copying qbc into device memory.

        (fastest-moving index first)
        PyClaw   order: (e, x, y)
        CUDAClaw order: (x, y, e)
        """

        cdef int err, meqn, mx, my

        qbc[:,num_ghost:-num_ghost,num_ghost:-num_ghost] = self.q

        meqn = qbc.shape[0]
        mx   = qbc.shape[1]
        my   = qbc.shape[2]

        qbct = np.rollaxis(qbc, 0, 3)
        qbct = np.asfortranarray(np.reshape(qbct, (mx, my, meqn), order='F'))

        self._set_qbc(qbct)

        self.synced = True

        return qbc

class CUDASolver2D(clawpack.pyclaw.ClawSolver2D):
    r"""  See the corresponding PyClaw Solver documentation."""

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def allocate_workspace(self, solution):
        r"""
        allocate_workspace (solution):

        Allocates memory for CUDA subroutines based on solution
        """

        state = solution.state
        grid  = state.grid

        cdef int ghostCells = self.num_ghost

        cdef int cellsX = grid.num_cells[0] + ghostCells*2
        cdef int cellsY = grid.num_cells[1] + ghostCells*2
        cdef int numStates = 3
        cdef int numWaves  = 3
        cdef int numCoeff  = 1
        cdef real startX = grid.x.lower
        cdef real endX   = grid.x.upper
        cdef real startY = grid.y.lower
        cdef real endY   = grid.y.upper
        cdef real startTime = 0
        cdef real endTime = 1e10

        shallow_water_solver_allocate(cellsX,
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
                                      endTime)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setup(self, solution):
        r"""
        setup (solution):

        Sets up a 2-D Shallow-Water Riemann solver with appropriate boundary
        conditions, limiter, grid size, and ghost cells.
        """

        cdef int err

        # PyClaw boundary conditions
        # bc_lower = [left, down]
        # bc_upper = [right, up]

        # CUDACLAW boundary conditions
        # bc = [left, right, up, down]
        # left, ... columns ... , right
        # up, ... rows ... , down

        bc_left  = self.bc_lower[0]
        bc_right = self.bc_upper[0]
        bc_down  = self.bc_lower[1]
        bc_up    = self.bc_upper[1]

        limiter  = self.limiters

        err = shallow_water_solver_setup(bc_left,
                                         bc_right,
                                         bc_up,
                                         bc_down,
                                         limiter)
        check_err(err)

        self.allocate_workspace(solution)
        self.allocate_bc_arrays(solution.state)

        self._is_set_up = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evolve_to_time(self,solution,tend=None):
        r"""
        Evolve solution from solution.t to tend.  If tend is not specified,
        take a single step.

        This method contains the machinery to evolve the solution object in
        ``solution`` to the requested end time tend if given, or one
        step if not.

        This method contains a subset of the functionality from the full
        evolve_to_time routine.

        :Input:
         - *solution* - (:class:`Solution`) Solution to be evolved
         - *tend* - (float) The end time to evolve to, if not provided then
           the method will take a single time step.

        :Output:
         - (dict) - Returns the status dictionary of the solver
        """

        if not self._is_set_up:
            self.setup(solution)

        if tend == None:
            take_one_step = True
        else:
            take_one_step = False

        # Parameters for time-stepping
        tstart = solution.t

        # Reset status dictionary
        self.status['cflmax'] = float('NaN')
        self.status['dtmin'] = self.dt
        self.status['dtmax'] = self.dt
        self.status['numsteps'] = 0

        # Setup for the run
        if self.dt_variable:
            raise Exception('Variable time steps currently disabled')

        if take_one_step:
            self.max_steps = 1
        else:
            self.max_steps = int((tend - tstart) / self.dt)
        if tend <= tstart and not take_one_step:
            self.logger.info("Already at or beyond end time: no evolution required.")
            self.max_steps = 0

        state = solution.state

        if not state.synced:
            self.qbc = state.get_qbc_from_q(self.num_ghost,self.qbc)

        # Main time-stepping loop
        for n in xrange(self.max_steps):

            # Adjust dt so that we hit tend exactly if we are near tend
            if not take_one_step:
                if solution.t + self.dt > tend and tstart < tend:
                    self.dt = tend - solution.t
                if tend - solution.t - self.dt < 1.e-14:
                    self.dt = tend - solution.t

            saved_dt = self.dt

            ### explicitly hard-code dt here

            self.step(solution)

            self.dt = saved_dt

            # Accept this step
            solution.t = tstart+(n+1)*self.dt
            # Verbose messaging
            self.logger.debug("Step %i  CFL = %f   dt = %f   t = %f"
                % (n,float('NaN'),self.dt,solution.t))

            self.write_gauge_values(solution)
            # Increment number of time steps completed
            self.status['numsteps'] += 1

            # See if we are finished yet
            if solution.t >= tend or take_one_step:
                break

        # End of main time-stepping loop -------------------------------------

        state = solution.state
        if not state.synced:
            state.set_q_from_qbc(self.num_ghost,self.qbc)

        return self.status


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def step(self, solution):
        """
        step (solution):

        Advances the 2-D Shallow-Water solution by the requested time dt,
        then returns a suggested next_dt that obeys the CFL condition for
        the current solution by a safety factor of 0.9.

        This method leaves the solution *unsynced* with CUDA device memory,
        the user must call set_q_from_qbc to retrieve the solution from the
        device.

        Unsupported functionality:

        solver.before_step
        solver.step_source
        solver.cfl

        :Input:
         - *solution* - (:class:`~cudaclaw.Solution`) solution to be evolved

        :Output:
         - (bool) - Currently always returns True (No CFL post-checks)

        """

        cdef real next_dt
        cdef real dt = self.dt

        err = hyperbolic_solver_2d_step(dt, &next_dt)
        check_err(err)

        solution.state.synced = False

        self.dt = next_dt
        return True
