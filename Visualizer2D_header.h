#ifndef __VISUALIZER2D_HEADER_H__
#define __VISUALIZER2D_HEADER_H__

#if defined (_WIN32) || defined (_WIN64)
#include <windows.h>
#endif

#include <iostream>

#include "common.h"
#include "glInfo.h"
#include <cuda_gl_interop.h>

#include <GL/gl.h>

#ifdef __APPLE__		// if Apple, macintosh
#include <GLUT/glut>
#else
#include <gl/glut.h>	// else if windows, or linux
#endif

#ifndef __GLUTINTERFACE_H__
#ifndef WINDOWS_GL_EXTENSIONS
#define WINDOWS_GL_EXTENSIONS
// function pointers for PBO Extension
// Windows needs to get function pointers from ICD OpenGL drivers,
// because opengl32.dll does not support extensions higher than v1.1.
#if defined (_WIN32) || defined (_WIN64)
PFNGLGENBUFFERSARBPROC pglGenBuffersARB = 0;                     // PBO Name Generation Procedure
PFNGLBINDBUFFERARBPROC pglBindBufferARB = 0;                     // PBO Bind Procedure
PFNGLBUFFERDATAARBPROC pglBufferDataARB = 0;                     // PBO Data Loading Procedure
PFNGLBUFFERSUBDATAARBPROC pglBufferSubDataARB = 0;               // PBO Sub Data Loading Procedure
PFNGLDELETEBUFFERSARBPROC pglDeleteBuffersARB = 0;               // PBO Deletion Procedure
PFNGLGETBUFFERPARAMETERIVARBPROC pglGetBufferParameterivARB = 0; // return various parameters of PBO
PFNGLMAPBUFFERARBPROC pglMapBufferARB = 0;                       // map PBO procedure
PFNGLUNMAPBUFFERARBPROC pglUnmapBufferARB = 0;                   // unmap PBO procedure
#define glGenBuffersARB           pglGenBuffersARB
#define glBindBufferARB           pglBindBufferARB
#define glBufferDataARB           pglBufferDataARB
#define glBufferSubDataARB        pglBufferSubDataARB
#define glDeleteBuffersARB        pglDeleteBuffersARB
#define glGetBufferParameterivARB pglGetBufferParameterivARB
#define glMapBufferARB            pglMapBufferARB
#define glUnmapBufferARB          pglUnmapBufferARB
#endif
#endif
#endif

//TODO: Add interactivity, or not
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////   Visualizer 2D Class  ////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Riemann_h, class Riemann_v, class Limiter, class BCS>
class Visualizer2D
{
public:
	// Gl info variable
	glInfo glInfo;

	// Problem variables
	pdeParam* param;
	BCS boundary_conditions;
	Riemann_h horizontal_solver;
	Riemann_v vertical_solver;
	Limiter limiter_function;

	// display variables
	// window
	int w_width;		// window widht
	int w_height;		// window height

	//canvas
	float canvas_ratio;	// width to height ratio, by convention the width is kept constant at 1
	float centerX, centerY, centerZ;

	// byte size of display, takes into account boundary cell display
	size_t displaySize;	// display size in Bytes

	// Problem Dimensions
	int dispResolutionX;			// includes main and ghost cells
	int dispResolutionY;			// includes main and ghost cells

	// display options
	int state_display;		// the state to be displayed
	bool boundary_display;	// to display or not the boundaries
	float range;			// display values will be normalized between -range and + range, not really an option

	// display control
	bool hold;				// toggles the computation
	bool step_once;			// allows to step a single step of computation
	bool stopDisplay;		// toggles the data copy to display operation
	bool colorScheme;		// rotates between grayscale and color scheme
	GLfloat intensity;

	// display capabilities
	bool supportPBO;
	bool supportNPOT;	// support for non power of two textures

	// interactivity
	//bool makeImpulse;	// interactivity flag, is set to true on a mouse click
	GLdouble impulseX, impulseY, impulseZ;

	// display and interoperability elements
	GLuint textureId;
	GLuint dispPBO;
	GLfloat* PBO_DISP_data_d;	// GPU resident
	cudaGraphicsResource_t PBO_DISP_CUDA_resource;

public:
	// Constructor / Destructor
	Visualizer2D(int window_width, int window_height, int dispResX, int dispResY);
	Visualizer2D();
	~Visualizer2D();

	// pdeParam member setting function
	inline void setParam(pdeParam &pdeParameters);
	inline void setBoundaryConditions(BCS conditions);
	inline void setSolvers(Riemann_h horizontalSolver, Riemann_v verticalSolver);
	inline void setLimiter(Limiter phiLimiter);

	// auxillary functions
	//float findAbsMax();
	//float setRange();
	inline void getOGLPos(int mouseX, int mouseY, GLdouble &coordX, GLdouble &coordY, GLdouble &coordZ);

	// GLUT initialization
	inline void InitGl();

	// The functions below initialize the necessary components for display
	inline void checkGPUCompatibility();
	inline void initializeDisplay();

	// main callback functions and copy function
	inline void visualizePDE();										// core of RENDER
	inline void drawCanvas();
	inline void checkChanges();
	inline void reshapeWindow(int w, int h);						// core of RESHAPE
	inline void normalKeyPress(unsigned char key, int x, int y);	// core of KEYPRESS
	inline void mouseClick(int button, int state, int x, int y);	// core of MOUSECLICK

	inline void update(double compute_time, double copy_time, double total_time);	// update

	// Display launcher
	inline void launchDisplay();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  Display Update  (signatures)//////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void copyDisplayData(GLfloat* PBO, int dispResolutionX, int dispResolutionY,				// DEPRECATED
								real* q, int cellsX, int cellsY, int numStates, int ghostCells,
								int state_display, bool boundary_display, GLfloat intensity);

extern "C" void copyDisplayData_Flat(GLfloat* PBO, int dispResolutionX, int dispResolutionY,
								real* q, int cellsX, int cellsY, int numStates, int ghostCells,
								int state_display, bool boundary_display, bool colorScheme, GLfloat intensity);
#endif