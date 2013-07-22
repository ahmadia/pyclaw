#!/usr/bin/env python
"""CUDAClaw

"""

import os
from os.path import join as pjoin
import subprocess
from distutils.core import setup
from distutils.extension import Extension
from numpy.distutils.core import setup
import numpy

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted by Robert from
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    This functionality should really be brought in from numpy or PyCUDA.

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib'
    and values giving the absolute path to each directory.

    Also returns 'cuflags', which allows for customizing compiler flags,
    these are currently hardcoded to 64-bit CUDA 5 flags

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib': pjoin(home, 'lib')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    cudaconfig['cuflags'] = '-m64 -gencode arch=compute_10,code=sm_10' + \
        ' -gencode arch=compute_20,code=sm_20' + \
        ' -gencode arch=compute_30,code=sm_30' + \
        ' -gencode arch=compute_35,code=sm_35'

    return cudaconfig

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

def configuration(parent_package='',top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration(
    	'cudaclaw',
    	parent_package,
    	top_path,
    	)
    config.add_data_files('log.config')
    config.add_subpackage('limiters')
    config.add_subpackage('io')
    config.add_extension("cudaclaw",
                         ["cudaclaw.pyx",
                          "cuda/cudaclaw.cu"],
                         language="c++",
                         library_dirs=[CUDA['lib']],
                         libraries=['cudart'],
                         runtime_library_dirs=[CUDA['lib']],
                         include_dirs=['cuda',numpy.get_include(),CUDA['include']])

    return config


if __name__ == '__main__':
    config = configuration(top_path='')
    setup(**config.todict())
