#ifndef __VISUALIZER2D_H__
#define __VISUALIZER2D_H__

#include "Visualizer2D_header.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////  Constructor/Destructor  ///////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::Visualizer2D()
{
	w_width = 512;
	w_height = 512;

	state_display = 0;
	boundary_display = false;

	hold = false;
	step_once = false;
	stopDisplay = false;
	colorScheme = true;
	force_continue = false;
	intensity = 1.0f;
	floor = 0.0f;
	ceil = 1.0f;
	//makeImpulse = false;

	PBO_DISP_CUDA_resource = NULL;
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::Visualizer2D(int window_width, int window_height, int dispResX, int dispResY)
{
	w_width = window_width;
	w_height = window_height;

	dispResolutionX = dispResX;
	dispResolutionY = dispResY;

	state_display = 0;
	boundary_display = true;
	
	hold = true;
	step_once = false;
	stopDisplay = false;
	colorScheme = true;
	force_continue = false;
	intensity = 1.0f;
	floor = 0.0f;
	ceil = 1.0f;
	//makeImpulse = false;

	PBO_DISP_CUDA_resource = NULL;
	// The rest of the members will be initialized
	// once the driver is set, and the opengl context launched
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::~Visualizer2D()
{
	param->clean();
	glDeleteTextures(1, &textureId);
	cudaGraphicsUnregisterResource(PBO_DISP_CUDA_resource);
	glDeleteBuffersARB(1, &dispPBO); // this does essentially: delete PBO_DISP_data_d
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////  OpenGL Setup/Launch  ////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::InitGl()
{
	glEnable(GL_DEPTH_TEST);    //Makes 3D drawing work when something is in front of something else
	glEnable(GL_NORMALIZE);		//Normalizes vectors that should be normalized
	glEnable(GL_TEXTURE_2D);	//Enables 2D textures
	glEnable(GL_CULL_FACE);		//Culls the back face by default

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	//glClearColor(0.2f,0.15f,0.2f,1.0f);	// Background color
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::initializeDisplay()
{
	// Initialize gl elements
	InitGl();

	// Initialize glew
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(1);
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	// Set Canvas ratio
	canvas_ratio = (float)(param->cellsX) / (float)(param->cellsY);
	centerX = 0.0f;
	centerY = 0.0f;
	centerZ = 0.0f;

	// Texture preparations
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // can be set to GL_NEAREST
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // can be set to GL_NEAREST
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // can be set to GL_LINEAR
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // can be set to GL_LINEAR
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glBindTexture(GL_TEXTURE_2D, 0);

	checkGPUCompatibility();

	// Do different things for different supports from the GPU
	// Find next largest power of two of both dimensions
	if (!supportNPOT)
	{
		//// Assuming non square texture is supported
		////1<<((int)min(ceil(log((double)DISPX)/log(2.0)), ceil(log(double(DISPY))/log(2.0))));
		//dispResolutionX = 1<<((int)ceil(log((double)dispResolutionX)/log(2.0)));
		//dispResolutionY = 1<<((int)ceil(log((double)dispResolutionY)/log(2.0)));

		//// if square texture is not supported, make it square
		//// dispResolutionX = min(displayResX, displayResY);
		//// dispResolutionY = dispResolutionX

		//printf("res x: %i, res y: %i\n", dispResolutionX, dispResolutionY);
		printf("Display will not be supported on this device!\n");
		exit(1);
	}
	else
	{
		if ( param->cellsX > MAX_RESOLUTION )			// We clamp the largest side to a maximum resolution size
		{												// and scale the rest accordingly
			if (canvas_ratio >= 1)
			{
				dispResolutionX = MAX_RESOLUTION;
				dispResolutionY = (int)(MAX_RESOLUTION/canvas_ratio);
			}
			else
			{
				dispResolutionY = MAX_RESOLUTION;
				dispResolutionX = (int)(MAX_RESOLUTION*canvas_ratio);
			}
		}
	}

	displaySize = dispResolutionX * dispResolutionY * sizeof(float)*3;	// Times 3 for RGB components

	//range = setRange();

	// WARNING! By this point we should have started an OpenGL context
	// Display PBO
	glGenBuffersARB(1, &dispPBO);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, dispPBO);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, displaySize, 0, GL_STREAM_DRAW_ARB);	// memory allocation for the buffer

	cudaError_t ret = cudaGraphicsGLRegisterBuffer(&PBO_DISP_CUDA_resource, dispPBO, cudaGraphicsMapFlagsWriteDiscard);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::reshapeWindow(int w, int h)
{
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero height).
	if (h == 0)
		h = 1;
	float ratio =  w * 1.0 / h;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45.0f, ratio, 0.1f, 100.0f);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
	gluLookAt(0.0,  0.0,   2.0,		//The camera position
			  0.0,  0.0,  -5.0,		//The point we're looking at
			  0.0f, 1.0f,  0.0f);	//The up vector
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::drawCanvas()
{
	glTranslatef(0.0, 0.0, 0.0);
	//canvas
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);

	glNormal3f(0.0f, 0.0f, 1.0f);

	// very simple dimension adjustment, will fix the width dimension to 1
	if (canvas_ratio >= 1)
	{
		glTexCoord2f(0.0f, 0.0f);												//
		glVertex3f(centerX-0.5f, centerY-(1.0f/canvas_ratio)/2.0f, centerZ);	// define
		// bottom
		glTexCoord2f(1.0f, 0.0f);												// line		//
		glVertex3f(centerX+0.5f, centerY-(1.0f/canvas_ratio)/2.0f, centerZ);	//			// define
		// right
		glTexCoord2f(1.0f, 1.0f);												//			// line
		glVertex3f(centerX+0.5f, centerY+(1.0f/canvas_ratio)/2.0f, centerZ);	// define	//
		// upper
		glTexCoord2f(0.0f, 1.0f);												// line
		glVertex3f(centerX-0.5f, centerY+(1.0f/canvas_ratio)/2.0f, centerZ);	//
	}
	else
	{
		glTexCoord2f(0.0f, 0.0f);												//
		glVertex3f(centerX-(canvas_ratio)/2.0f, centerY-0.5f, centerZ);			// define
		// bottom
		glTexCoord2f(1.0f, 0.0f);												// line		//
		glVertex3f(centerX+(canvas_ratio)/2.0f, centerY-0.5f, centerZ);			//			// define
		// right
		glTexCoord2f(1.0f, 1.0f);												//			// line
		glVertex3f(centerX+(canvas_ratio)/2.0f, centerY+0.5f, centerZ);			// define	//
		// upper
		glTexCoord2f(0.0f, 1.0f);												// line
		glVertex3f(centerX-(canvas_ratio)/2.0f, centerY+0.5f, centerZ);			//
	}

	glEnd();
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::launchDisplay()
{
	glutMainLoop();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////  Problem Setup  ////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::setParam(pdeParam &pdeParameters)
{
	param = &pdeParameters;
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::setBoundaryConditions(BCS conditions)
{
	boundary_conditions = conditions;
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::setSolvers(Riemann_h horizontalSolver, Riemann_v verticalSolver)
{
	horizontal_solver = horizontalSolver;
	vertical_solver = verticalSolver;
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::setLimiter(Limiter phiLimiter)
{
	limiter_function = phiLimiter;
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::setEntropy(Ent_h entropy_fix_horizontal, Ent_v entropy_fix_vertical)
{
	entropy_fix_h = entropy_fix_horizontal;
	entropy_fix_v = entropy_fix_vertical;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////  OpenGL Interactivity  ////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::getOGLPos(int mouseX, int mouseY, GLdouble &coordX, GLdouble &coordY, GLdouble &coordZ)
{
	GLint viewport[4];
	GLdouble modelview[16];
	GLdouble projection[16];
	GLfloat winX, winY, winZ;
	//GLdouble posX, posY, posZ;	//////////////////////////////////////////////////warning declared but never referenced. Why were they here?

	glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
	glGetDoublev( GL_PROJECTION_MATRIX, projection );
	glGetIntegerv( GL_VIEWPORT, viewport );

	winX = (GLfloat)mouseX;
	winY = (GLfloat)viewport[3] - (GLfloat)mouseY;
	glReadPixels( mouseX, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ );

	gluUnProject( winX, winY, winZ, modelview, projection, viewport, &coordX, &coordY, &coordZ);
}
//float Visualizer2D<Riemann_h, Riemann_v, Limiter>::findAbsMax(real* data, int width, int height, int paddedWidth, int numBoundCellsX, int numBoundCellsY)
//{
//	//// we have a choice to include or not the boundary cells in search for the absolute max, of the values
//	//// The default will be kept as including the boundary cells.
//	//float maximum = -FLT_MAX;
//
//	//for( int j = 0; j < height-1; j++)
//	//{
//	//	for( int i = 0; i < width; i++)
//	//	{
//	//		if (absoluteValue(data[i + j*paddedWidth]) > maximum)
//	//		{
//	//			maximum = absoluteValue(data[i + j*paddedWidth]);
//	//		}
//	//	}
//	//}
//	//return maximum;
//}
//float Visualizer2D<Riemann_h, Riemann_v, Limiter>::setRange()
//{
//	//Float * data_cpu = (Float*)malloc(paddedWidth*height*sizeof(Float));
//
//	//cudaMemcpy(data_cpu, driver->params.gpuQ, paddedWidth*height*sizeof(Float), cudaMemcpyDeviceToHost);	// This is inefficient but it is done once.
//
//	//float halfRange = findAbsMax(data_cpu, width, height, paddedWidth, driver->params.numBoundaryCellsX, driver->params.numBoundaryCellsY);
//
//	//free(data_cpu);
//
//	//if (halfRange == 0)
//	//	return 0.5;
//	//else
//	//	return halfRange;
//}
//extern "C" void doImpulse(GLdouble impulseX, GLdouble impulseY, GLdouble impulseZ,
//							pdeParam &param, Visualizer2D &vis);

template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::normalKeyPress(unsigned char key, int x, int y)
{
	if (key == 27)	// escape
	{
		gracefulExit();
		//param->clean();
		//glDeleteTextures(1, &textureId);
		//cudaGraphicsUnregisterResource(PBO_DISP_CUDA_resource);
		//glDeleteBuffersARB(1, &dispPBO); // this does essentially: delete PBO_DISP_data_d
		//
		//cudaThreadExit();
		//exit(0);
	}
	int maxNumStates = param->numStates;
	switch(key)
	{
	case '1':
		if (maxNumStates > 0 ) state_display = 0; break;
	case '2':
		if (maxNumStates > 1 ) state_display = 1; break;
	case '3':
		if (maxNumStates > 2 ) state_display = 2; break;
	case '4':
		if (maxNumStates > 3 ) state_display = 3; break;
	case '5':
		if (maxNumStates > 4 ) state_display = 4; break;
	case '6':
		if (maxNumStates > 5 ) state_display = 5; break;
	case '7':
		if (maxNumStates > 6 ) state_display = 6; break;
	case '8':
		if (maxNumStates > 7 ) state_display = 7; break;
	case '9':
		if (maxNumStates > 8 ) state_display = 8; break;
	case '0':
		if (maxNumStates > 9 ) state_display = 9; break;

	case 'b':
	case 'B':
		boundary_display = !boundary_display; break;

	case 'c':
	case 'C':
		stopDisplay = !stopDisplay; break;
	
	case 'f':
	case 'F':
		force_continue = true;break;

	case 'g':
	case 'G':
		colorScheme = !colorScheme; break;

	case 'h':
	case 'H':
		hold = !hold; break;

	case 's':
	case 'S':
		if (hold)
			step_once = true;
		else
			hold = true;
		break;

	case '[':
		floor -= 0.2f; break;
	case '}':
		floor += 0.2f; break;

	case ']':
		ceil += 0.2f; break;
	case '{':
		ceil -= 0.2f; break;

	case '\\':
		floor = 0.0f;
		ceil  = 1.0f;
		break;

	case '+':
		intensity += 0.02f;
		break;

	case '-':
		intensity += -0.02f;
		break;

	case '*':
		intensity = 1.0f;
		break;

	default:
		break;
	}
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, class BCS, class Ent_h, class Ent_v>::mouseClick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		getOGLPos(x, y, impulseX, impulseY, impulseZ);

		if (button == GLUT_LEFT_BUTTON)
		{
			//makeImpulse = true;
		}
	}
	else // if (state == GLUT_UP)
	{
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  GPU Capability  //////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Check for PBO and NPOT texture support on the video card
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v,  Limiter, BCS, Ent_h, Ent_v>::checkGPUCompatibility()
{
	glInfo.getInfo();

	supportPBO = false;
	supportNPOT = false;
#if defined (_WIN32) || defined (_WIN64)
	// check PBO is supported by your video card
	if(glInfo.isExtensionSupported("GL_ARB_pixel_buffer_object"))
	{
		// get pointers to GL functions
		glGenBuffersARB		=		(PFNGLGENBUFFERSARBPROC)wglGetProcAddress("glGenBuffersARB");
		glBindBufferARB		=		(PFNGLBINDBUFFERARBPROC)wglGetProcAddress("glBindBufferARB");
		glBufferDataARB		=		(PFNGLBUFFERDATAARBPROC)wglGetProcAddress("glBufferDataARB");
		glBufferSubDataARB	=		(PFNGLBUFFERSUBDATAARBPROC)wglGetProcAddress("glBufferSubDataARB");
		glDeleteBuffersARB	=		(PFNGLDELETEBUFFERSARBPROC)wglGetProcAddress("glDeleteBuffersARB");
		glGetBufferParameterivARB = (PFNGLGETBUFFERPARAMETERIVARBPROC)wglGetProcAddress("glGetBufferParameterivARB");
		glMapBufferARB		=		(PFNGLMAPBUFFERARBPROC)wglGetProcAddress("glMapBufferARB");
		glUnmapBufferARB	=		(PFNGLUNMAPBUFFERARBPROC)wglGetProcAddress("glUnmapBufferARB");

		// check once again PBO extension
		if(glGenBuffersARB && glBindBufferARB && glBufferDataARB && glBufferSubDataARB &&
			glMapBufferARB && glUnmapBufferARB && glDeleteBuffersARB && glGetBufferParameterivARB)
		{
			supportPBO = true;
			cout << "Video card supports GL_ARB_pixel_buffer_object." << endl;
		}
		else
		{
			supportPBO = false;
			cout << "Video card does NOT support GL_ARB_pixel_buffer_object." << endl;
		}
	}
	if (glInfo.isExtensionSupported("GL_ARB_texture_non_power_of_two"))
	{
		supportNPOT = true;
		cout << "Video card supports GL_ARB_texture_non_power_of_two." << endl;
	}
#else // for linux, do not need to get function pointers, it is up-to-date
	if(glInfo.isExtensionSupported("GL_ARB_pixel_buffer_object"))
	{
		supportPBO = true;
		cout << "Video card supports GL_ARB_pixel_buffer_object." << endl;
	}
	else
	{
		supportPBO = false;
		cout << "Video card does NOT support GL_ARB_pixel_buffer_object." << endl;
	}
	if (glInfo.isExtensionSupported("GL_ARB_texture_non_power_of_two"))
	{
		supportNPOT = true;
		cout << "Video card supports GL_ARB_texture_non_power_of_two." << endl;
	}
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////  Main Loop  ////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::visualizePDE()
{
	static real simulation_time = (real)0.0f;
	static real simulation_stepTime = (real)0.0f;
	static real simulation_timestamp = (real)0.0f;
	Timer mainLoopTimer;
	Timer stepWatch;
	Timer copyWatch;
	mainLoopTimer.start();

	checkChanges();	// check for any interactivity flag and apply changes, reset necessary flags to default.

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,  0.0,   2.0,		//The camera position
			  0.0,  0.0,  -5.0,		//The point we're looking at
			  0.0f, 1.0f,  0.0f);	//The up vector

	/***********************************************Display and Update***********************************************/
	
	// snapshot control
	if (param->snapshots)
	{
		static int snap_number = 1;

		if( snap_number == 1 && param->snapshotTimeInterval == (real)0.0f )
			param->takeSnapshot(0, "pde data");	// take initial state snapshot

		static real simulationTimeInterval = (real) 0.0f;

		simulationTimeInterval += simulation_stepTime;
		if (simulationTimeInterval > param->snapshotTimeInterval)
		{
			param->takeSnapshot(snap_number, "pde data");
			simulationTimeInterval = (real) 0.0f;
			snap_number++;
		}
	}

	if (!stopDisplay && simulation_timestamp >= (real)0.001f )		// simulation view speed, we view the simulation every x (0.001) seconds, TODO: make this a controllable variable
	{
		copyWatch.start();
		cudaGraphicsMapResources(1, &PBO_DISP_CUDA_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&PBO_DISP_data_d, &displaySize, PBO_DISP_CUDA_resource);

		// This function can be modified to put the read data into a format that can be used for colored mapping
		copyDisplayData(PBO_DISP_data_d, dispResolutionX, dispResolutionY,
						param->qNew, param->cellsX, param->cellsY, param->numStates, param->ghostCells,
						state_display, boundary_display, colorScheme, intensity, floor, ceil);/**/

		// The below would be useable if double texture is supported, and boundaries are to be included:		// turns out not really, when padding is on
		// Not just padding, different sizes of resolution and data size...
		//cudaMemcpy(PBO_DISP_data_d, param->qNew, displaySize, cudaMemcpyDeviceToDevice);

		cudaGraphicsUnmapResources(1, &PBO_DISP_CUDA_resource, 0);
		simulation_time += simulation_timestamp;
		simulation_timestamp = (real)0.0f;
		copyWatch.stop();
	}

	glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, dispPBO);
	glBindTexture(GL_TEXTURE_2D, textureId); //Tell OpenGL which texture to edit, I only need to do this once, and it is done in the intialisations

	glTexImage2D(	GL_TEXTURE_2D,
					0,
					GL_RGB,				//GL_LUMINANCE,	before colors
					dispResolutionX,	//(disp_width+2*driver->params.numBoundaryCellsX),
					dispResolutionY,	//(disp_height+2*driver->params.numBoundaryCellsY),
					0,
					GL_RGB,				//GL_LUMINANCE, before colors
					GL_FLOAT,
					0);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
	if ( (!hold || (hold && step_once)) && (simulation_time < param->endTime || force_continue) )
	{
			stepWatch.start();
			simulation_stepTime = step(*param, horizontal_solver, vertical_solver, limiter_function, boundary_conditions, entropy_fix_h, entropy_fix_v);
			simulation_timestamp += simulation_stepTime;
			//cudaThreadSynchronize();
			stepWatch.stop();
	}
	if (step_once)
	{
		step_once = false;
	}

	//***************************************************************************************************************//

	drawCanvas();

	glutSwapBuffers();
	
	mainLoopTimer.stop();

	update(stepWatch.getElapsedTimeInMilliSec(), copyWatch.getElapsedTimeInMilliSec(), mainLoopTimer.getElapsedTimeInMilliSec());	// pass the simulation time to this to display
}
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
inline void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::checkChanges()
{/*
	if (makeImpulse)
	{	
		
		doImpulse(impulseX, impulseY, impulseZ, centerX, centerY, centerZ, canvas_ratio,
			driver->params.gpuQ,
			driver->params.calculateXCells(),	// returns padded number
			width,								// returns unpadded number
			height,
			driver->params.numBoundaryCellsX,
			driver->params.numBoundaryCellsY,
			state_display);					// given the impulse coords the deimensions of the canvas, the given state
		*/
		/*
		// suggestion, make inside visualizer2D.h under canvas,
		a center for the canvas and the width and hight of the canvas,
		later in init display we can assign values to these and use them in draw canvas,
		we can then use these values and the impulse coordinates to determine
		the exact location where the impulse should be, mainly by getting the ration of where the hit was,
		and finding the most appropriate location corresponding to it
		in the computational data and making a kind of impulse there.
		*/
	/*
		makeImpulse = false;
	}*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////  Timers  /////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Riemann_h, class Riemann_v, class Limiter, class BCS, class Ent_h, class Ent_v>
void Visualizer2D<Riemann_h, Riemann_v, Limiter, BCS, Ent_h, Ent_v>::update(double compute_time, double copy_time, double total_time)
{
	static float aggregate_compute_time = 0.0;
	static float aggregate_copy_time = 0.0;
	static float aggregate_total_time = 0.0;

	static int frames = 0;
	frames++;

	aggregate_compute_time += compute_time;
	aggregate_copy_time    += copy_time;
	aggregate_total_time   += total_time;

	// TODO: have things display inside the window and Not the title
	if (aggregate_total_time > 250.0)	// when a quarter of a second has passed
	{
		float stepTimeAverage = aggregate_compute_time / frames;
		float copyTimeAverage = aggregate_copy_time    / frames;
		float fps             = 1000.0 * frames / aggregate_total_time;

		char title[256];

		sprintf(title, "2D Visuals of PDE Solution, State: %d -FPS: %7f ", state_display + 1, fps);
		
		if (hold)
			sprintf(title, "%s-Compute stopped ", title);
		else
			sprintf(title, "%s-Step: %7fms, ", title, stepTimeAverage);

		if (stopDisplay)
			sprintf(title, "%s-Display stopped ", title);
		else
			sprintf(title, "%s-Copy: %7fms", title, copyTimeAverage);

		glutSetWindowTitle(title);

		frames = 0;
		aggregate_compute_time = 0.0;
		aggregate_copy_time    = 0.0;
		aggregate_total_time   = 0.0;
	}
	glutPostRedisplay(); //Tell GLUT that the scene has changed
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////// Visualiser2D.cu
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Some getter functions similiar to those in pdeParam struct
// Row major order, not used
inline __device__ int getIndex_q(int cellsX, int cellsY, int numStates, int row, int column, int state)
{
	// Usual C/C++ row major order
	return (row*cellsX*numStates + column*numStates + state);
}
inline __device__ real &getElement_q(real* q, int cellsX, int cellsY, int numStates, int row, int column, int state)
{
	return q[getIndex_q(cellsX, cellsY, numStates, row, column, state)];
}

// Flat version of the getter functions, that is the state is the slowest moving dimension here
inline __device__ int getIndexFlat_q(int cellsX, int cellsY, int numStates, int row, int column, int state)
{
	// state is now the slowest moving dimension, followed by row then column
	return (state*cellsX*cellsY + row*cellsX + column);
}
inline __device__ real &getElementFlat_q(real* q, int cellsX, int cellsY, int numStates, int row, int column, int state)
{
	return q[getIndexFlat_q(cellsX, cellsY, numStates, row, column, state)];
}

// PBO getters and setters and memory access pattern
inline __device__ int getIndex_PBO(int dispResolutionX, int dispResolutionY, int row, int column)
{
	// Usual C/C++ row major order
	return (row*dispResolutionX*3 + column*3);
}
inline __device__ GLfloat &getElement_PBO(GLfloat* PBO, int dispResolutionX, int dispResolutionY, int row, int column)
{
	return PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column)];
}
inline __device__ void setDisplayPBO_grayScale(GLfloat* PBO, int dispResolutionX, int dispResolutionY, int row, int column, GLfloat pixelValue)
{
	PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 0] = pixelValue;	// R
	PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 1] = pixelValue;	// G
	PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 2] = pixelValue;	// B
}
inline __device__ void setDisplayPBO_color(GLfloat* PBO, int dispResolutionX, int dispResolutionY, int row, int column, GLfloat pixelValue)
{
	// Jet color scheme, Matlab
	if ( pixelValue <= 0 )
	{
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 0] = 0.0f;		// R
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 1] = 0.0f;		// G
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 2] = 0.458333f;	// B
	}
	else if ( pixelValue >= 1 )
	{
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 0] = 0.458333f;	// R
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 1] = 0.0f;		// G
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 2] = 0.0f;		// B
	}
	else
	{
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 0] = 1.5f - fabs(4.1667f*(pixelValue-0.75f));	// R
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 1] = 1.5f - fabs(4.1667f*(pixelValue-0.50f));	// G
		PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column) + 2] = 1.5f - fabs(4.1667f*(pixelValue-0.25f));	// B
	}
}

/*
float findAbsMaxPerFrame(Float* data, int width, int height, int paddedWidth, int numBoundCellsX, int numBoundCellsY)
{
	// we have a choice to include or not the boundary cells in search for the absolute max, of the values
	// The default will be kept as including the boundary cells.
	float maximum = -FLT_MAX;

	for( int j = 0; j < height-1; j++)
	{
		for( int i = 0; i < width; i++)
		{
			if (absoluteValue(data[i + j*paddedWidth]) > maximum)
			{
				maximum = absoluteValue(data[i + j*paddedWidth]);
			}
		}
	}
	printf("\nmax of this frame is: %f.\n\n", maximum);
	return maximum;
}*/
/*
float setRangePerFrame(Float * data_gpu, int width, int paddedWidth, int height, int numBoundCellsX, int numBoundCellsY)
{
	Float * data_cpu = (Float*)malloc(paddedWidth*height*sizeof(Float));

	cudaMemcpy(data_cpu, data_gpu, paddedWidth*height*sizeof(Float), cudaMemcpyDeviceToHost);	// This is inefficient but it is done once.

	float halfRange = findAbsMaxPerFrame(data_cpu, width, height, paddedWidth, numBoundCellsX, numBoundCellsY);

	free(data_cpu);

	if (halfRange == 0)
		return 0.5;//1.0f;
	else
		return halfRange;
}*/

__global__ void copyDisplay_Kernel(GLfloat* PBO, int dispResolutionX, int dispResolutionY,
									real* q, int cellsX, int cellsY, int numStates, int ghostCells,
									int state_display, bool boundary_display, bool colorScheme, GLfloat intensity, GLfloat floor, GLfloat ceil)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	float cell_disp_ratioX = (float)(cellsX)/(float)dispResolutionX;
	float cell_disp_ratioY = (float)(cellsY)/(float)dispResolutionY;

	if ( col < dispResolutionX && row < dispResolutionY )
	{
		float sum = 0.0f;
		int cells = 0;
		for (int i = 0; i < cell_disp_ratioY; i++)
		{
				for (int j = 0; j < cell_disp_ratioX; j++)
				{
					sum += getElementFlat_q(q, cellsX, cellsY, numStates, (int)(row*cell_disp_ratioY) + i, (int)(col*cell_disp_ratioX) + j, state_display);
					cells++;
				}
		}
		GLfloat range = ceil-floor;
		GLfloat average = ((sum/cells)-floor)/range;
		
		if (colorScheme)
		{
			setDisplayPBO_color(PBO, dispResolutionX, dispResolutionY, row, col, intensity*average);

			if (!boundary_display)
			{
				if (col < ghostCells || col >= dispResolutionX - ghostCells)
					setDisplayPBO_color(PBO, dispResolutionX, dispResolutionY, row, col, 0.0f);
				else if (row < ghostCells || row >= dispResolutionY - ghostCells)
					setDisplayPBO_color(PBO, dispResolutionX, dispResolutionY, row, col, 0.0f);
			}
		}
		else
		{
			setDisplayPBO_grayScale(PBO, dispResolutionX, dispResolutionY, row, col, intensity*average);

			if (!boundary_display)
			{
				if (col < ghostCells || col >= dispResolutionX - ghostCells)
					setDisplayPBO_grayScale(PBO, dispResolutionX, dispResolutionY, row, col, 0.0f);
				else if (row < ghostCells || row >= dispResolutionY - ghostCells)
					setDisplayPBO_grayScale(PBO, dispResolutionX, dispResolutionY, row, col, 0.0f);
			}
		}
	}
}

extern "C" void copyDisplayData(GLfloat* PBO, int dispResolutionX, int dispResolutionY,
								real* q, int cellsX, int cellsY, int numStates, int ghostCells,
								int state_display, bool boundary_display, bool colorScheme, GLfloat intensity, GLfloat floor, GLfloat ceil)
{
	// this function would be more interesting when GPU does not support non power of two textures //?

	// kernel to copy the data from vis.param.qNew to vis.PBO_DISP_data_d
	unsigned int blockDimensionX = 32;
	unsigned int blockDimensionY = 16;

	unsigned int gridDimensionX = (dispResolutionX+blockDimensionX-1)/blockDimensionX;
	unsigned int gridDimensionY = (dispResolutionY+blockDimensionY-1)/blockDimensionY;

	dim3 dimGrid(gridDimensionX, gridDimensionY);
	dim3 dimBlock(blockDimensionX, blockDimensionY);

	//range = setRangePerFrame(data, width, paddedWidth, height, numBoundCellsX, numBoundCellsY); //?

	copyDisplay_Kernel<<<dimGrid, dimBlock>>>(PBO, dispResolutionX, dispResolutionY,
												q, cellsX, cellsY, numStates, ghostCells,
												state_display, boundary_display, colorScheme, intensity, floor, ceil);
}

/*
__global__ void pulse_Kernel(int X, int Y,
							 Float * data, int padded_width, int width, int height, int numBoundCellsX, int numBoundCellsY, int state = 0)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;

	// this is quite innefficient in terms of warps being scheduled and divergence,
	// but seeing that it happens only by user input it can be overlooked
	if ( (tidx - X)*(tidx - X) + (tidy - Y)*(tidy - Y) < (float)min(width,height)/4.0)
	{
		data[state*(padded_width*height) + tidy*(padded_width) + tidx] = 
			data[state*(padded_width*height) + tidy*(padded_width) + tidx]
			+ 10.0f*exp( -1.0f*((float)(tidx - X)*(tidx - X)/width + (float)(tidy - Y)*(tidy - Y))/height );
	}
}
*/
/*
extern "C" void doImpulse(GLdouble impulseX, GLdouble impulseY, GLdouble impulseZ,
						  float centerX, float centerY, float centerZ, float canvasRatio,
						  Float * data, int paddedWidth, int width, int height, int numBoundCellsX, int numBoundCellsY,
						  int state)
{
	// check if impulse coordinate is on canvas, checking only the Z should be enough
	if ( absoluteValue(impulseZ - centerZ) < 0.001 )
	{
		// do whatever ratio thing that has to be done, and do some kind of impulse.
		float xRatio = (float)(impulseX - centerX) + 0.5f; // the width is fixed at 1, centered around 0, from -0.5 to 0.5
		float yRatio = ((float)(impulseY - centerY) + 0.5f/canvasRatio)*canvasRatio;

		int approxX = xRatio*width;
		int approxY = yRatio*height;

		//// reading values at point click
		//Float clickedValue = 0.0;
		//cudaMemcpy(&clickedValue, (data+ state*paddedWidth*height + approxY*paddedWidth + approxX), sizeof(Float), cudaMemcpyDeviceToHost);
		//float clickedValue_float = clickedValue;
		//
		//printf("\n************************** Value of clicked point is: %f\n", clickedValue_float);

		if ( xRatio < 0.9 && yRatio < 0.9 && xRatio > 0.1 && yRatio > 0.1 )
		{
			// kernel to copy the data from driver->params.gpuQ or driver->params.gpuQNew to PBO_DISP_data_d
			unsigned int blockDimensionX = 16;
			unsigned int blockDimensionY = 16;

			unsigned int gridDimensionX = (width+blockDimensionX-1)/blockDimensionX;
			unsigned int gridDimensionY = (height+blockDimensionY-1)/blockDimensionY;

			dim3 dimGrid(gridDimensionX, gridDimensionY);
			dim3 dimBlock(blockDimensionX, blockDimensionY);

			pulse_Kernel<<<dimGrid, dimBlock>>>(approxX, approxY, data, paddedWidth, width, height, numBoundCellsX, numBoundCellsY, state);
		}
	}
}

*/

#endif	// VISUALIZER2D