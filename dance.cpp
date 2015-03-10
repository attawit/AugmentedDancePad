// Skeleton Code for CS290I Homework 1
// 2012, Jon Ventura and Chris Sweeney

// adapt the include statements for your system:

#include "background_segm.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>


#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include <cstdio>
#include <fstream>

#define NUM_CELLS 4
#define PATTERN_COL_RATIO 0.25
#define PATTERN_TIME_UNIT 0.25
#define ARROW_SIDE 40.0

using namespace cv;
using namespace std;

VideoCapture *cap = new VideoCapture(0);
int width = 1280;
int height = 720;
Mat image;


Mat background_image;
vector< vector<int> > patterns;
float timer_start = 0.0;

// pattern display
Mat arrow_left, arrow_right, arrow_up, arrow_down;
double start_line_padding = 50.0;
double finish_line_padding = 50.0;


// boolean flag
bool start_play = false;

// read dance patterns from the file
void read_patterns(const char* file_name);

// background subtraction
void background_subtraction();

// plane detection
void plane_detection();

// get timer
float timer();

// start the pattern
void start_pattern(Mat& image);

// overlay the image
void overlay(Mat& foreground, Mat& background, double x, double y, double width, double height);

// a useful function for displaying your coordinate system
void drawAxes(float length)
{
    glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT) ;
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) ;
    glDisable(GL_LIGHTING) ;
    
    glBegin(GL_LINES) ;
    glColor3f(1,0,0) ;
    glVertex3f(0,0,0) ;
    glVertex3f(length,0,0);
    
    glColor3f(0,1,0) ;
    glVertex3f(0,0,0) ;
    glVertex3f(0,length,0);
    
    glColor3f(0,0,1) ;
    glVertex3f(0,0,0) ;
    glVertex3f(0,0,length);
    glEnd() ;
    
    
    glPopAttrib() ;
}

void overlay(Mat& foreground, Mat& background, int x, int y, int width, int height){
    for (int i = y; i < y+height; i++){
        for(int j = x; j < x+width; j++){
            double opacity =((double)foreground.data[(i-y) * foreground.step + (j-x) * foreground.channels() + 3])/ 255.;
            for(int c = 0; opacity > 0 && c < background.channels(); ++c)
            {
                unsigned char foregroundPx = foreground.data[(i-y) * foreground.step + (j-x) * foreground.channels() + c];
                unsigned char backgroundPx = background.data[i * background.step +  j * background.channels() + c];
                background.data[i*background.step + background.channels()*j + c] = backgroundPx * (1.-opacity) + foregroundPx * opacity;
            }
        }
    }
}

void start_pattern(Mat& image){
    for (int i = 1; i <= NUM_CELLS ; i++) {
        line(image, Point(image.cols - PATTERN_COL_RATIO*image.cols*i/(NUM_CELLS+1), 0), Point(image.cols - PATTERN_COL_RATIO*image.cols*i/(NUM_CELLS+1), image.rows), Scalar(255, 0, 0));
    }
    
    line(image, Point(image.cols, start_line_padding), Point(image.cols - PATTERN_COL_RATIO*image.cols, start_line_padding), Scalar(255, 0, 0));
    line(image, Point(image.cols, image.rows-finish_line_padding), Point(image.cols - PATTERN_COL_RATIO*image.cols, image.rows-finish_line_padding), Scalar(255, 0, 0));
    
    float time_now = timer();
//    cout << "time now: " << time_now << endl;
    // for each line of the pattern
    for (int i = 0; i < patterns.size(); i++) {
        float distance = (time_now - i * PATTERN_TIME_UNIT) * (ARROW_SIDE*8);
        
        // if it's not the turn to show up for the rest, just break the loop
        if (distance < 0)
            break;
        
        // if it's already finished, just skip it to conitue
        if (distance > image.rows - start_line_padding - finish_line_padding) {
            continue;
        }
        
        if (patterns[i][0] == 1) {
            // because it's the unflip image, we use arrow_right to indicate the arrow to the left
            overlay(arrow_right, image, (int)(image.cols-PATTERN_COL_RATIO*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
        if (patterns[i][1] == 1) {
            overlay(arrow_up, image, (int)(image.cols-PATTERN_COL_RATIO*2*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
        if (patterns[i][2] == 1) {
            overlay(arrow_down, image, (int)(image.cols-PATTERN_COL_RATIO*3*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
        if (patterns[i][3] == 1) {
            // because it's the unflip image, we use arrow_left to indicate the arrow to the right
            overlay(arrow_left, image, (int)(image.cols-PATTERN_COL_RATIO*4*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
    }
}


void display()
{
    // clear the window
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    
    cv::Mat tempimage;
    flip(image, tempimage, -1);
    
    line(tempimage, Point(0, 0), Point(0, image.rows), Scalar(255, 0, 0));
    line(tempimage, Point(PATTERN_COL_RATIO*image.cols, 0), Point(PATTERN_COL_RATIO*image.cols, image.rows), Scalar(255, 0, 0));
    
    
    glDisable(GL_DEPTH_TEST);
    glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    glEnable(GL_DEPTH_TEST);
    
    
    
    //////////////////////////////////////////////////////////////////////////////////
    // Here, set up new parameters to render a scene viewed from the camera.
    
    //set viewport
    glViewport(0, 0, tempimage.size().width, tempimage.size().height);
    
    //set projection matrix using intrinsic camera params
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    
    //you will have to set modelview matrix using extrinsic camera params
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    
    
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);
    
    
    /////////////////////////////////////////////////////////////////////////////////
    // Drawing routine
    
    //now that the camera params have been set, draw your 3D shapes
    //first, save the current matrix
   
    
    glTranslatef(0, 0, 0);
    //drawAxes(1.0);
    
    
    // show the rendering on the screen
    glutSwapBuffers();
    
    // post the next redisplay
    glutPostRedisplay();
}

void reshape( int w, int h )
{
    // set OpenGL viewport (drawable area)
    glViewport( 0, 0, w, h );
}

void mouse( int button, int state, int x, int y )
{
    if ( button == GLUT_LEFT_BUTTON && state == GLUT_UP )
    {
        
    }
}

void keyboard( unsigned char key, int x, int y )
{
    clock_t t = clock();
    switch ( key )
    {
        case 'q':
            // quit when q is pressed
            exit(0);
            break;
        case 's':
            start_play = !start_play;
            // adjust the the base line of the timer
            timer_start = float(t)/CLOCKS_PER_SEC;
            break;
        default:
            break;
    }
}

void idle()
{
    // grab a frame from the camera
    (*cap) >> image;
    
    if (image.empty()) {
        cout << "No captured frame" << endl;
        exit(0);
    }
    
    if (start_play) {
        start_pattern(image);
    }
}

void read_patterns(const char* file_name){
    fstream infile(file_name, ios_base::in);
    string temp;
    while (getline(infile, temp))
    {
        istringstream buffer(temp);
        int number;
        vector<int> pattern;
        for (int i = 0; i < NUM_CELLS; i++){
            buffer >> number;
            pattern.push_back(number);
        }
        patterns.push_back(pattern);
    }
    
    cout << "pattern: " << endl;
    for (int i = 0; i < patterns.size(); i++) {
        cout << patterns[i][0] << ", " << patterns[i][1] << ", " << patterns[i][2] << ", " << patterns[i][3] << endl;
    }
}

float timer(){
    clock_t t = clock();
    float now = float(t)/CLOCKS_PER_SEC;
    return now-timer_start;
}


int main( int argc, char **argv )
{
    int w,h;
    
    if ( argc == 1 ) {
        // start video capture from camera
        cap = new cv::VideoCapture(0);
    } else if ( argc == 2 ) {
        // start video capture from file
        cap = new cv::VideoCapture(argv[1]);
    } else {
        fprintf( stderr, "usage: %s [<filename>]\n", argv[0] );
        return 1;
    }
    
    // check that video is opened
    if ( cap == NULL || !cap->isOpened() ) {
        fprintf( stderr, "could not start video capture\n" );
        return 1;
    }
    
    // get width and height
    w = (int) cap->get( CV_CAP_PROP_FRAME_WIDTH );
    h = (int) cap->get( CV_CAP_PROP_FRAME_HEIGHT );
    // On Linux, there is currently a bug in OpenCV that returns
    // zero for both width and height here (at least for video from file)
    // hence the following override to global variable defaults:
    width = w ? w : width;
    height = h ? h : height;

    
    read_patterns("./patterns.txt");
    Mat arrow_up_tmp, arrow_down_tmp, arrow_left_tmp, arrow_right_tmp;
    arrow_left_tmp = imread("./image/arrow_left.png", CV_LOAD_IMAGE_COLOR);
    arrow_right_tmp = imread("./image/arrow_right.png", CV_LOAD_IMAGE_COLOR);
    arrow_up_tmp = imread("./image/arrow_up.png", CV_LOAD_IMAGE_COLOR);
    arrow_down_tmp = imread("./image/arrow_down.png", CV_LOAD_IMAGE_COLOR);
    resize(arrow_left_tmp, arrow_left, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_right_tmp, arrow_right, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_up_tmp, arrow_up, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_down_tmp, arrow_down, Size(ARROW_SIDE, ARROW_SIDE));
    
//    cap->open(0);
//    cap->set(CV_CAP_PROP_FPS, 100.0);
    // initialize GLUT
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA| GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition( 20, 20 );
    glutInitWindowSize( width, height );

    glutCreateWindow( "Augmented Dance Pad" );
    
    
    // set up GUI callback functions
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glDepthFunc(GL_LESS);
    glShadeModel(GL_SMOOTH);
    glDisable(GL_CULL_FACE); // Nicer looking teapots.
    glClearColor(0.0f, 0.0f, 0.0f, 1);
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0f);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glDisable(GL_COLOR_MATERIAL);
    
    
    
    

    glutDisplayFunc( display );
    glutReshapeFunc( reshape );
    glutMouseFunc( mouse );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    
    // start GUI loop
    glutMainLoop();
    
    delete cap;
    return 0;
}
