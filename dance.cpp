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

using namespace cv;
using namespace std;

cv::VideoCapture *cap = NULL;
int width = 1280;
int height = 720;
Mat image; //original image read in from video stream

//parameters for background substraction
Mat background_image; 
cv::BackgroundSubtractorMOG2 bgs;  
bool background_image_flag = false;

//parameters for calibration
bool plane_detection_flag = false;
vector<Point3f>  objectPoints;
Size boardSize;
float squareSize = 5;
vector<Point2f> pointBuf;
bool found = false;
cv::Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
cv::Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
Mat rotation;
Mat translation;
Mat rotationRodrigues;
float modelview[4][4];
float projection[4][4];
float zNear = 0.5;
float zFar = 500;


// read dance patterns from the file
void read_patterns(const char* file_name);

// background subtraction
void background_subtraction();

// plane detection
void plane_detection();

//read in camera calibration parameters
void readParameters(const char* fileName);

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

void initialiseOpenGL(){
  glClearColor(0.0, 0.0, 0.0, 1.0);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();   
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();   

  glShadeModel(GL_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING);
  glEnable(GL_NORMALIZE);

}

void showCorners()
{
    GLUquadricObj *qobj = gluNewQuadric();
    glPushMatrix();
    
    for(int i = 0; i<8; i++){
      for(int j=0; j<6; j++){
        glPopMatrix();
        glPushMatrix();
        //glScalef(-1,-1,1);
        glTranslatef(squareSize*10, 0, 0);
        glTranslatef(squareSize*i, 0, squareSize*j);
        gluSphere(qobj,1,16,16);
      }
    }
    glPopMatrix();
}

void display()
{
    // clear the window
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    cv::Mat tempimage;
    //flip(image, tempimage, 0);
    image.copyTo(tempimage);
    if(background_image_flag){
        cvtColor( background_image, tempimage, CV_GRAY2BGR );
        //flip(tempimage,tempimage, -1);

    }
        //flip(background_image, tempimage, 0);
    
    
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
    if(found){
        glLoadMatrixf(*projection);
    }

    //you will have to set modelview matrix using extrinsic camera params
    glMatrixMode(GL_MODELVIEW);
    if(found){
        glLoadIdentity();
        glLoadMatrixf(*modelview);
        //glScalef(1.0,-1.0,-1.0);
    }
    else{
        glLoadIdentity();
    }
    gluLookAt(0, 0, 0, 0, 0, -5, 0, 1, 0);  

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    
    
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);
    
    
    /////////////////////////////////////////////////////////////////////////////////
    // Drawing routine
    
    if(found){
     glPushMatrix();
      //move to the position where you want the 3D object to go
      glTranslatef(0, 0, 0); //this is an arbitrary position for demonstration
      //you will need to adjust your transformations to match the positions where
      //you want to draw your objects(i.e. chessboard center, chessboard corners)
      glRotatef(90, -1.0, 0.0, 0.0);
      
      drawAxes(squareSize);
      showCorners();

    glPopMatrix();
    //glClear(GL_DEPTH_BUFFER_BIT);
  }
    
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
    switch ( key )
    {
        case 'q':
            // quit when q is pressed
            exit(0);
            break;
        case 'b':
            if(background_image_flag)
                background_image_flag = false;
            else
                background_image_flag = true;
            break;
        default:
            break;
    }
}

void idle()
{
    // grab a frame from the camera
    (*cap) >> image;
    flip(image, image, -1);
    if (image.empty()) {
        cout << "No captured frame" << endl;
        exit(0);
    }

    background_subtraction(); 

    plane_detection();
}



int main( int argc, char **argv )
{
    int w,h;
    boardSize.height = 6;
    boardSize.width = 8;

    if ( argc == 2 ) {
        // start video capture from camera
        cap = new cv::VideoCapture(0);
    } else if ( argc == 3 ) {
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

    //initialize background subtractor
    bgs.nmixtures = 3;
    bgs.history = 1000;
    bgs.varThresholdGen = 15;
    bgs.bShadowDetection = true;                            
    bgs.nShadowDetection = 0;                               
    bgs.fTau = 0.5;  
    
    readParameters(argv[1]);//read in camera calibration parameters
    for( int i = 0; i < boardSize.height; ++i )
        for( int j = 0; j < boardSize.width; ++j )
          objectPoints.push_back(Point3f(float( j*squareSize ), float( i*squareSize ), 0));

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
    
    return 0;
}

// background subtraction
void background_subtraction(){
    Mat blurred;
    cv::GaussianBlur(image,blurred,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);
    Mat fg; //foreground
    bgs.operator()(blurred,fg);
    Mat bgmodel;       
    bgs.getBackgroundImage(bgmodel); 
    Mat medianed; 
    medianBlur(fg, medianed, 5); //median filtering it                        
    cv::threshold(medianed,background_image,50.0f,255,CV_THRESH_BINARY);
}

void plane_detection(){

  found = findChessboardCorners( image, boardSize, pointBuf,
                CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK);
  //cout<<found<<endl;
  
  if(found){
    Mat temp = image.clone();
    undistort(temp, image, cameraMatrix, distCoeffs);
    
    //get extrinsic matrix
    
    solvePnP(Mat(objectPoints), Mat(pointBuf), cameraMatrix, distCoeffs, rotationRodrigues, translation,false,CV_ITERATIVE);
    Rodrigues(rotationRodrigues, rotation);

    vector<Point2f> pointBuf2;
    /*cout<<"object: "<<Mat(objectPoints)<<endl;
    cout<<"rotation: "<<Mat(rotation)<<endl;
    cout<<"rotation3: "<<rotationRodrigues<<endl;
    cout<<"trans: "<<Mat(translation)<<endl;
    cout<<"ip "<<Mat(pointBuf)<<endl;
    cout<<"camera matrix: "<<cameraMatrix<<endl;
    cout<<"distortion: "<<distCoeffs<<endl;
*/
    projectPoints(Mat(objectPoints), rotationRodrigues, Mat(translation), cameraMatrix, distCoeffs, pointBuf2);
    //drawChessboardCorners( image, boardSize, Mat(pointBuf2), found );
    //set the projection matrix
    projection[0][0] = 2*cameraMatrix.at<double>(0,0)/width;
    projection[0][1] = 0;
    projection[0][2] = 0;
    projection[0][3] = 0;

    projection[1][0] = 0;
    projection[1][1] = -2*cameraMatrix.at<double>(1,1)/height;
    projection[1][2] = 0;
    projection[1][3] = 0;

    projection[2][0] = 1-2*cameraMatrix.at<double>(0,2)/width;
    projection[2][1] = 1-(2*cameraMatrix.at<double>(1,2)+2)/height;
    projection[2][2] = (zNear+zFar)/(zNear - zFar);
    projection[2][3] = -1;

    projection[3][0] = 0;
    projection[3][1] = 0;
    projection[3][2] = 2*zNear*zFar/(zNear - zFar);
    projection[3][3] = 0;
 
    //set the modelview matrix
    modelview[0][0] = rotation.at<double>(0,0);
    modelview[0][1] = -rotation.at<double>(1,0);
    modelview[0][2] = -rotation.at<double>(2,0);
    modelview[0][3] = 0;

    modelview[1][0] = rotation.at<double>(0,1);
    modelview[1][1] = -rotation.at<double>(1,1);
    modelview[1][2] = -rotation.at<double>(2,1);
    modelview[1][3] = 0;

    modelview[2][0] = rotation.at<double>(0,2);
    modelview[2][1] = -rotation.at<double>(1,2);
    modelview[2][2] = -rotation.at<double>(2,2);
    modelview[2][3] = 0;

    modelview[3][0] = translation.at<double>(0,0);
    modelview[3][1] = -translation.at<double>(1,0);
    modelview[3][2] = -translation.at<double>(2,0);
    modelview[3][3] = 1;

    }
}

void readParameters(const char* fileName){
  ifstream inFile;
  inFile.open(fileName);
  if(!inFile.is_open()){
    cout<<"Could not open camera.txt"<<endl;
  }
  
  char buffer[30];
  inFile.getline(buffer,30);

  inFile >> cameraMatrix.at<double>(0,0);
  inFile >>  cameraMatrix.at<double>(0,1);
  inFile >>  cameraMatrix.at<double>(0,2);
  inFile >>  cameraMatrix.at<double>(1,0);
  inFile >>  cameraMatrix.at<double>(1,1);
  inFile >>  cameraMatrix.at<double>(1,2);
  inFile >>  cameraMatrix.at<double>(2,0);
  inFile >>  cameraMatrix.at<double>(2,1);
  inFile >>  cameraMatrix.at<double>(2,2);
  char buffer2[30];
  inFile.getline(buffer2,30); 
  inFile.getline(buffer2,30); 
  inFile.getline(buffer2,30);
  inFile >>  distCoeffs.at<double>(0,0);
  inFile >>  distCoeffs.at<double>(1,0);
  inFile >>  distCoeffs.at<double>(2,0);
  inFile >>  distCoeffs.at<double>(3,0);
  inFile >>  distCoeffs.at<double>(4,0);
}
