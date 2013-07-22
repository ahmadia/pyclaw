#ifndef __GLUTINTERFACE_H__
#define __GLUTINTERFACE_H__


template<class Vis>
struct GlutInterface
{
	static Vis* visualizer;

	static void VisualizerConstructor(int window_width, int window_height, int dispResX, int dispResY);

	static void InitGlut(int &argc, char** argv, int window_width, int window_height, int dispResX, int dispResY);

	~GlutInterface();

	// static functions to be used as callback functions
	static void RESHAPE(int w, int h);
	static void RENDER();
	static void KEYPRESS(unsigned char key, int x, int y);
	static void MOUSECLICK(int button, int state, int x, int y);
};

template<class Vis>
GlutInterface<Vis>::~GlutInterface()
{
	delete visualizer;
}

template<class Vis>
inline void GlutInterface<Vis>::RESHAPE(int w, int h)
{
	visualizer->reshapeWindow(w, h);
}
template<class Vis>
inline void GlutInterface<Vis>::RENDER()
{
	visualizer->visualizePDE();
}
template<class Vis>
inline void GlutInterface<Vis>::KEYPRESS(unsigned char key, int x, int y)
{
	visualizer->normalKeyPress(key, x, y);
}
template<class Vis>
inline void GlutInterface<Vis>::MOUSECLICK(int button, int state, int x, int y)
{
	visualizer->mouseClick(button, state, x, y);
}
template<class Vis>
void GlutInterface<Vis>::VisualizerConstructor(int window_width, int window_height, int dispResX, int dispResY) // pass args
{
	// check if already initialized
	visualizer = new Vis(window_width, window_height, dispResX, dispResY);
}
template<class Vis>
void GlutInterface<Vis>::InitGlut(int &argc, char** argv, int window_width, int window_height, int dispResX, int dispResY)
{
	VisualizerConstructor(window_width, window_height, dispResX, dispResY);

	// Initialise Glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);

	// Create and position window
	glutInitWindowPosition(100,100);
	glutInitWindowSize(window_width,window_height);
	glutCreateWindow("2D Visuals of PDE Solution");

	// register callbacks
	//
	// Resize function
	glutReshapeFunc(RESHAPE);

	// Set functions for drawing
	glutDisplayFunc(RENDER);

	// Normal key press handler
	glutKeyboardFunc(KEYPRESS);

	// Mouse click handler
	glutMouseFunc(MOUSECLICK);
}

#endif