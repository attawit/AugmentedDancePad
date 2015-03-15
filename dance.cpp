// Skeleton Code for CS290I Homework 1
// 2012, Jon Ventura and Chris Sweeney

// adapt the include statements for your system:

#include <string>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <thread>
//#include <boost/thread.hpp>
#include "helper.h"

//#define PATTERN_FILE "pattern.txt"
//#define MUSIC_FILE "music.wav"
#define BLOCK_SIZE 50
#define VELOCITY_THRESHOLD 5.0
#define HAS_OBJ_RATIO_THRESHOLD 0.2

using namespace cv;
using namespace std;

const char* PATTERN_FILE = "pattern.txt";
const char* MUSIC_FILE = "music.wav";

VideoCapture *cap;
int width = 1280;
int height = 720;
Mat image; //original image read in from video stream
const char snap_name[] = "snapshot.bmp";

//parameters for background substraction
Mat background_image; 
cv::BackgroundSubtractorMOG2 bgs;  
bool background_image_flag = false;

//parameters for calibration
bool plane_detection_flag = true;
vector<Point3f>  objectPoints;
Size boardSize;
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


// boolean flag
bool start_play = false;

bool move_front = false;
bool move_back = false;
bool move_left = false;
bool move_right = false;
bool stop_front = false;
bool stop_back = false;
bool stop_left = false;
bool stop_right = false;

//ROI of the dance pad's blocks
Point2f front_bl, front_ur, left_bl, left_ur, right_bl, right_ur, back_bl, back_ur;

// music
thread music_thread;
thread::id music_thread_id;
char* music_file_path;


/** Declaration **/
// the thread function to play music
void play_music();

// draw dance pad
void drawDancePad();

// background subtraction
void background_subtraction();

// plane detection
void plane_detection();



/** Implementation **/

void play_music(){
    std::this_thread::sleep_for (std::chrono::seconds(5));
    std::cout << "pause of " << TIME_PREPARATION << " seconds ended\n";
    system("afplay ./music/jmww/music.wav");
}

void drawDancePad()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);
    GLUquadricObj *qobj = gluNewQuadric();
    glPushMatrix();

    //left   
    glTranslatef(squareSize*10, 0, -squareSize*2);
    glColor4f(1.0, 153.0/255.0, 153.0/255.0,0.5); 
    glBegin(GL_POLYGON); 
    glVertex3f(0.0, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, BLOCK_SIZE); 
    glVertex3f(0, 0.0, BLOCK_SIZE); 
    glEnd(); 
    
    //right
    glTranslatef(2*BLOCK_SIZE, 0, 0);
    glColor4f(204.0/255.0, 1.0, 153.0/255.0,0.5); 
    glBegin(GL_POLYGON); 
    glVertex3f(0.0, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, BLOCK_SIZE); 
    glVertex3f(0, 0.0, BLOCK_SIZE); 
    glEnd(); 
    //up
    glTranslatef(-BLOCK_SIZE, 0, BLOCK_SIZE);
    glColor4f(1.0, 1.0, 153.0/255.0,0.5); 
    glBegin(GL_POLYGON); 
    glVertex3f(0.0, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, BLOCK_SIZE); 
    glVertex3f(0, 0.0, BLOCK_SIZE); 
    glEnd();
    //down
    glTranslatef(0, 0, -2*BLOCK_SIZE);
    glColor4f(153.0/255.0, 204.0/255.0, 1.0,0.5); 
    glBegin(GL_POLYGON); 
    glVertex3f(0.0, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, BLOCK_SIZE); 
    glVertex3f(0, 0.0, BLOCK_SIZE); 
    glEnd();

    glPopMatrix();

    GLdouble tx, ty, tz;
    GLdouble _modelview[16], _projection[16];
    GLint _viewport[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, _modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, _projection);
    glGetIntegerv(GL_VIEWPORT, _viewport);
    gluProject(.0, .0, .0, _modelview, _projection, _viewport, &tx, &ty, &tz);

    //ROI of the left block
    gluProject(squareSize*10, 0, -squareSize*2+BLOCK_SIZE*0.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    left_bl.x = tx;
    gluProject(squareSize*10, 0, -squareSize*2, _modelview, _projection, _viewport, &tx, &ty, &tz);
    left_bl.y = ty;
    gluProject(squareSize*10+BLOCK_SIZE, 0, -squareSize*2+BLOCK_SIZE*0.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    left_ur.x = tx;
    gluProject(squareSize*10+BLOCK_SIZE, 0, -squareSize*2+BLOCK_SIZE, _modelview, _projection, _viewport, &tx, &ty, &tz);
    left_ur.y = ty;

    //ROI of the right block
    gluProject(squareSize*10+BLOCK_SIZE*2, 0, -squareSize*2+BLOCK_SIZE*0.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    right_bl.x = tx;
    gluProject(squareSize*10+BLOCK_SIZE*2, 0, -squareSize*2, _modelview, _projection, _viewport, &tx, &ty, &tz);
    right_bl.y = ty;
    gluProject(squareSize*10+BLOCK_SIZE*3, 0, -squareSize*2+BLOCK_SIZE*0.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    right_ur.x = tx;
    gluProject(squareSize*10+BLOCK_SIZE*3, 0, -squareSize*2+BLOCK_SIZE, _modelview, _projection, _viewport, &tx, &ty, &tz);
    right_ur.y = ty;

    //ROI of the front block
    gluProject(squareSize*10+BLOCK_SIZE, 0, -squareSize*2-BLOCK_SIZE*0.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    front_bl.x = tx;
    gluProject(squareSize*10+BLOCK_SIZE, 0, -squareSize*2-BLOCK_SIZE, _modelview, _projection, _viewport, &tx, &ty, &tz);
    front_bl.y = ty;
    gluProject(squareSize*10+BLOCK_SIZE*2, 0, -squareSize*2-BLOCK_SIZE*0.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    front_ur.x = tx;
    gluProject(squareSize*10+BLOCK_SIZE*2, 0, -squareSize*2, _modelview, _projection, _viewport, &tx, &ty, &tz);
    front_ur.y = ty;

    //ROI of the back block
    gluProject(squareSize*10+BLOCK_SIZE, 0, -squareSize*2+BLOCK_SIZE*1.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    back_bl.x = tx;
    gluProject(squareSize*10+BLOCK_SIZE, 0, -squareSize*2+BLOCK_SIZE, _modelview, _projection, _viewport, &tx, &ty, &tz);
    back_bl.y = ty;
    gluProject(squareSize*10+BLOCK_SIZE*2, 0, -squareSize*2+BLOCK_SIZE*1.5, _modelview, _projection, _viewport, &tx, &ty, &tz);
    back_ur.x = tx;
    gluProject(squareSize*10+BLOCK_SIZE*2, 0, -squareSize*2+BLOCK_SIZE*2, _modelview, _projection, _viewport, &tx, &ty, &tz);
    back_ur.y = ty;

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

Point2f get_velocity(Point2f rect_bl, Point2f rect_ur){
    return front_bl;
}


bool hasObjInRoi(Point2f rect_bl, Point2f rect_ur){
    Mat mat(background_image, Rect(round(rect_bl.x), round(rect_bl.y), round(rect_ur.x), round(rect_ur.y)));
    // count how many pixels is filled
    int count = countNonZero(mat);
    // if the count is greater than some ratio of the whole roi,
    // we say it contains object
    if (count > (rect_ur.x-rect_bl.x)*(rect_ur.y-rect_bl.y) * HAS_OBJ_RATIO_THRESHOLD)
        return true;
    else
        return false;
}

void motionDetectionHelper(bool& pre, bool& stop, Point2f rect_bl, Point2f rect_ur){
    Point2f velocity = get_velocity(rect_bl, rect_ur);

    if (pre) {
        if (!hasObjInRoi(rect_bl, rect_ur)) {
            // if there is no object in the roi, set pre_fron to false for the next frame
            pre = false;
        }else{
            if (velocity.y < 0) {
                if (get_vel_length(velocity) > VELOCITY_THRESHOLD) {
                    // if it's moving, set pre_front for the next frame
                    pre = true;
                }else{
                    pre = false;
                    stop = true;
                }
            }else{
                pre = false;
            }
        }
    }else{
        // if the velocity in the y axis is equal to or greater than 0, than the sub-velocity is to the upside, just discard it.
        if (velocity.y < 0) {
            if (get_vel_length(velocity) > VELOCITY_THRESHOLD) {
                // if it's moving, set pre_front for the next frame
                pre = true;
            }
        }
    }
}

void motionDetection(){
    // detect whether there is movement following a stop in each roi
    motionDetectionHelper(move_front, stop_front, front_bl, front_ur);
    motionDetectionHelper(move_back, stop_back, back_bl, back_ur);
    motionDetectionHelper(move_left, stop_left, left_bl, left_ur);
    motionDetectionHelper(move_front, stop_front, front_bl, front_ur);
    
    
    
    
}

void display()
{
    // clear the window
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    
    cv::Mat tempimage;
    image.copyTo(tempimage);
    line(tempimage, Point(0, 0), Point(0, image.rows), Scalar(255, 0, 0));
    line(tempimage, Point(PATTERN_COL_RATIO*image.cols, 0), Point(PATTERN_COL_RATIO*image.cols, image.rows), Scalar(255, 0, 0));
    //flip(image, tempimage, 0);
    
    if(background_image_flag){
        cvtColor( background_image, tempimage, CV_GRAY2BGR );
        //flip(tempimage,tempimage, -1);
    }
        //flip(background_image, tempimage, 0);
    
    
    glDisable(GL_DEPTH_TEST);
    //glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    //alpha blend test start
    if(background_image_flag){
        
        Mat overlay_test ;//=  Mat::zeros(image.size(), CV_8UC3);
        Mat chans[3]; //
        split(image,chans);
        Mat newchans[4];
        newchans[0] = chans[0];
        newchans[1] = chans[1];
        newchans[2] = chans[2];
        newchans[3] = Mat::ones(image.rows, image.cols, CV_8UC1);///background_image;
        merge(newchans,4,overlay_test);
        
        //show the 2D block ROI position
        rectangle(overlay_test, left_bl, left_ur,cvScalar(0, 255, 255, 100), CV_FILLED, 8, 0);
        rectangle(overlay_test, right_bl, right_ur,cvScalar(0, 255, 255, 100), CV_FILLED, 8, 0);
        rectangle(overlay_test, front_bl, front_ur,cvScalar(0, 255, 255, 100), CV_FILLED, 8, 0);
        rectangle(overlay_test, back_bl, back_ur,cvScalar(0, 255, 255, 100), CV_FILLED, 8, 0);
        glDrawPixels( overlay_test.size().width, overlay_test.size().height, GL_BGRA, GL_UNSIGNED_BYTE, overlay_test.ptr() );
        
    }else{
        glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    }
    //alpha blend test end
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
      //showCorners();
      drawDancePad();

    glPopMatrix();
    //glClear(GL_DEPTH_BUFFER_BIT);
  }
    

    glDisable(GL_DEPTH_TEST);
    //glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    //alpha blend test start
    if(background_image_flag){
        glBlendFunc(GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA);
        Mat overlay_test ;//=  Mat::zeros(image.size(), CV_8UC3);
        Mat chans[3]; //
        Mat temp;
        split(image,chans);
        Mat newchans[4];
        newchans[0] = chans[0];
        newchans[1] = chans[1];
        newchans[2] = chans[2];
        //cv::threshold(background_image,temp,50.0f,1,CV_THRESH_BINARY);
        temp = cv::Scalar::all(255) - background_image;
        newchans[3] = temp;//Mat::zeros(image.rows, image.cols, CV_8UC1);///background_image;
        merge(newchans,4,overlay_test);
        glDrawPixels( overlay_test.size().width, overlay_test.size().height, GL_BGRA, GL_UNSIGNED_BYTE, overlay_test.ptr() );
   
    }else{
        glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    }
    //alpha blend test end
    glEnable(GL_DEPTH_TEST);



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
            music_thread = thread(play_music);
            music_thread_id = music_thread.get_id();
            cout << music_thread_id << endl;
            start_pattern(image);
            break;
        case 'b':
            if(background_image_flag)
                background_image_flag = false;
            else
                background_image_flag = true;
            break;
        case 'c':
            if(plane_detection_flag)
                plane_detection_flag = false;
            else
                plane_detection_flag = true;
            break;
        case 'x':
            snapshot(image.size().width,image.size().height,snap_name);
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
    if(plane_detection_flag){
       plane_detection(); 
    }
    
    if (start_play) {
        start_pattern(image);
    }
}

int main( int argc, char **argv )
{
    int w,h;
    boardSize.height = 6;
    boardSize.width = 8;
    
    if ( argc == 3 ) {
        // start video capture from camera
        cap = new VideoCapture(0);
    } else {
        fprintf( stderr, "usage: %s <calibration file> <music directory>\n", argv[0] );
        return 1;
    }
    
    // check that video is opened
    if ( cap == NULL || !cap->isOpened() ) {
        fprintf( stderr, "Could not start video capture\n" );
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
    
    //initialize pattern reading
    readPatterns(argv[2], PATTERN_FILE);
    
    //read arrow images
    readArrows(arrow_up, arrow_down, arrow_left, arrow_right);
    
    //read music
    music_file_path = new char[100];
    readMusic(argv[2], music_file_path, MUSIC_FILE);
    
    //initialize background subtractor
    bgs.nmixtures = 5;
    bgs.history = 1000;
    bgs.varThresholdGen = 15;
    bgs.bShadowDetection = true;
    bgs.nShadowDetection = 0;
    bgs.fTau = 0.5;
    
    readParameters(argv[1], cameraMatrix, distCoeffs);//read in camera calibration parameters
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
    // glEnable(GL_LIGHTING);
    // glEnable(GL_LIGHT0);
    glDisable(GL_COLOR_MATERIAL);
    
    
    glutDisplayFunc( display );
    glutReshapeFunc( reshape );
    glutMouseFunc( mouse );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    
    // start GUI loop
    glutMainLoop();
    
    music_thread.join();
    delete[] music_file_path;
    return 0;
}

