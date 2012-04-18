def test_3d_acoustics():
    """ test_3d_acoustics """

    def acoustics_verify_homogeneous(return_tuple):
        from clawpack.pyclaw.util import check_diff

        test_final_difference = return_tuple[1]
        return check_diff(0.00286, test_final_difference, abstol=1e-4)

    def acoustics_verify_heterogeneous(return_tuple):
        import os
        import numpy as np
        from clawpack.pyclaw.util import check_diff

        test_pfinal = return_tuple[0]
        thisdir = os.path.dirname(__file__)
        verify_pfinal = np.loadtxt(os.path.join(thisdir,'verify_classic_heterogeneous.txt'))
        norm_err = np.linalg.norm(test_pfinal-verify_pfinal)
        return check_diff(0, norm_err, abstol=1e-4)

    from clawpack.pyclaw.util import gen_variants
    from acoustics import acoustics3D

    homogeneous_tests   = gen_variants(acoustics3D, acoustics_verify_homogeneous,
                                       python_kernel=False, solver_type='classic', test='homogeneous')

    heterogeneous_tests = gen_variants(acoustics3D, acoustics_verify_heterogeneous,
                                       python_kernel=False, solver_type='classic', test='heterogeneous')
    
    from itertools import chain
    for test in chain(homogeneous_tests, heterogeneous_tests):
        yield test