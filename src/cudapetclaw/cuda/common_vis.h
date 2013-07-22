#ifndef __COMMON_VIS_H__
#define __COMMON_VIS_H__

//#define GLUT_NO_LIB_PRAGMA	// disable automatic glut linking

// Effects things in the visualizer class only
// determines the maximum resolution per side of the texture
#define MAX_RESOLUTION 512

#include "common.h"

#define GLEW_STATIC
#include <gl/glew.h>

#include <cuda_gl_interop.h>

#ifdef __APPLE__		// if Apple, macintosh
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <glut.h>
#else
#include <GL/gl.h>
#include <gl/glut.h>	// else if windows, or linux
#endif


#endif //-Xptxas -dlcm=cg 