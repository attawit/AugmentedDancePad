// Skeleton Code for CS290I Homework 1
// 2012, Jon Ventura and Chris Sweeney

// adapt the include statements for your system:

#include <string>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <thread>
#include "helper.h"

//#define PATTERN_FILE "pattern.txt"
//#define MUSIC_FILE "music.wav"
#define BLOCK_SIZE 50

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


//ROI of the dance pad's blocks
Point2f front_bl, front_ur, left_bl, left_ur, right_bl, right_ur, back_bl, back_ur;

//detection of movement
bool stop_front = false;
bool stop_back = false;
bool stop_left = false;
bool stop_right = false;

//detection threshold
float mag_threshold = 0.08;

//optical flow parameters
bool needToInit = false;
vector<Point2f> points[2];
const int MAX_COUNT = 500;
TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
Size subPixWinSize(10,10), winSize(31,31);
Mat gray,prevGray;
bool show_of_flag = false;
Mat previous;
float grad2deg = 180.0f/3.1415927;
Mat colored_of; //colored optical flow
int of_threshold = 15;

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
    system("afplay ./music/jmww/music.wav");
}

void snapshot(int windowWidth, int windowHeight, const char* filename){
  cv::Mat img(height, width, CV_8UC3);
  //use fast 4-byte alignment (default anyway) if possible
  glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);

  //set length of one complete row in destination data (doesn't need to equal img.cols)
  glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
  glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);
  Mat flipped;
  cv::flip(img, flipped, 0);
  //imwrite(filename, flipped);
  IplImage* temp;
  temp = cvCreateImage(cvSize(flipped.cols, flipped.rows),8,3);
  IplImage ipltemp = flipped;
  cvCopy(&ipltemp, temp);
  cvSaveImage(filename, temp);
}

void findLK(Mat image){
    //LK optical flow
    cvtColor(image, gray, COLOR_BGR2GRAY);

    if(needToInit){
         // automatic initialization
        
        Rect userROI(Point2f(front_bl.x,front_ur.y),Point2f(right_bl.x,back_ur.y));
        //cout<<"ROI"<<userROI<<endl;
        Mat mask = Mat::zeros(gray.size(), CV_8UC1);  // type of mask is CV_8U
        cout<<"size "<<gray.size()<<endl;
        mask(userROI) = 1; 
       //   cout<<"generate point single passenger"<<endl;
        goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, mask, 3, 0, 0.04);
            //cout<<points[1]<<endl;
        cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            //addRemovePt = false;
    }else if(!points[0].empty()){
        //    cout<<"draw point single passenger"<<endl;
            //cout<<points[1]<<endl;
        vector<uchar> status;
        vector<float> err;
        if(prevGray.empty())
            gray.copyTo(prevGray);
        calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                             3, termcrit, 0, 0.001);
        size_t i, k;
        for( i = k = 0; i < points[1].size(); i++ )
        {
            if( !status[i] )
                continue;
                points[1][k++] = points[1][i];
                circle( image, points[1][i], 3, Scalar(255,255,0), -1, 8);
        }
        points[1].resize(k);
    }
    
    needToInit = false;
    std::swap(points[1], points[0]);
    cv::swap(prevGray, gray);

}

bool withinRect(Point2f p, Point2f bl, Point2f ur){
    if(p.x>bl.x&&p.x<ur.x&&p.y>bl.y&&p.y<ur.y)
        return true;
    else
        return false;
}

//draw colorful optical flow image
void FlowToRGB(const cv::Mat & inpFlow, cv::Mat & rgbFlow){
  float max_size =-1.0f;
  bool use_value = true;
  float mean_val = 0, min_val = 1000, max_val = 0;
  float _dx, _dy;
  float sum_x, sum_y = 0.0f;
  Mat flow_magnitude = Mat(inpFlow.rows,inpFlow.cols, CV_32FC1);
  Mat flow_angle = Mat(inpFlow.rows,inpFlow.cols, CV_32FC1);
  //cv::Mat xy[2]; //X,Y
  //cv::split(inpFlow, xy);
  //cv::Mat magnitude, angle;
  //cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);
  //magnitude.convertTo(magnitude, CV_8UC1);
  //angle.convertTo(angle, CV_8UC1);
  //double mag_max, angle_max;
  //cv::minMaxLoc(magnitude, 0, &mag_max,0,0);
  //cv::minMaxLoc(angle, 0, &angle_max,0,0);
  //cout<<"flow size: "<<inpFlow.size()<<" COLS: "<<inpFlow.rows<<endl;
  for(int r = 0; r < flow_magnitude.rows; r++)
  {
      for(int c = 0; c < flow_magnitude.cols; c++)
      {
        mean_val += flow_magnitude.at<float>(r,c);
        
        Vec2f flow_at_point = inpFlow.at<Vec2f>(r, c);
        flow_magnitude.at<float>(r,c) = norm(flow_at_point);
        flow_angle.at<float>(r,c) = (atan(flow_at_point[1]/flow_at_point[0])*grad2deg+180.0f);
        //sum_x += flow_at_point[0]/1000.0f;
        //sum_y += flow_at_point[1]/1000.0f;
        max_val = MAX(max_val, flow_magnitude.at<float>(r,c));
        min_val = MIN(min_val, flow_magnitude.at<float>(r,c));
        //cout<<flow_angle.at<float>(r,c)<<" ";

      }
  }
  
  mean_val /= inpFlow.size().area();
  //sum_x = sum_x/(float)(flow_magnitude.rows)/(float)(flow_magnitude.cols);
  //sum_y = sum_y/(float)(flow_magnitude.rows)/(float)(flow_magnitude.cols);
  double scale = max_val - min_val;
  double shift = -min_val;//-mean_val + scale;
  scale = 255.f/scale;
  if( max_size > 0)
  {
      scale = 255.f/max_size;
      shift = 0;
  }
  //cout<<"max: "<<max_val<<" min: "<<min_val<<" scale: "<<scale<<"sum_x "<<sum_x<<"sum_y "<<sum_y<<endl;
  //calculate the angle, motion value 
  cv::Mat hsv(flow_magnitude.size(), CV_8UC3);
  uchar * ptrHSV = hsv.ptr<uchar>();
  int idx_val  = (use_value) ? 2:1;
  int idx_sat  = (use_value) ? 1:2;

  for(int r = 0; r < inpFlow.rows; r++, ptrHSV += hsv.step1())
  {
      uchar * _ptrHSV = ptrHSV;
      for(int c = 0; c < inpFlow.cols; c++, _ptrHSV+=3)
      {
          // cv::Point2f vpol = pol.at<cv::Point2f>(r,c);

          // _ptrHSV[0] = cv::saturate_cast<uchar>(vpol.x);
          // _ptrHSV[idx_val] = cv::saturate_cast<uchar>( (vpol.y + shift) * scale);  
          // _ptrHSV[idx_sat] = 255;
        _ptrHSV[0] = cv::saturate_cast<uchar>(flow_angle.at<float>(r,c));
        _ptrHSV[idx_val] = cv::saturate_cast<uchar>( (flow_magnitude.at<float>(r,c) + shift) * scale);  
        //fixed scaling
        //_ptrHSV[idx_val] = cv::saturate_cast<uchar>( (flow_magnitude.at<float>(r,c) ) * 30.0f);  

        //ptrHSV[idx_val] = cv::saturate_cast<uchar>( magnitude.at<double>(r,c));  
        //cout<<angle.at<double>(r,c)<<" ";
        _ptrHSV[idx_sat] = 255;
      }
  }   
  cv::Mat rgbFlow32F;
  cv::cvtColor(hsv, rgbFlow32F, CV_HSV2BGR);
  rgbFlow32F.convertTo(rgbFlow, CV_8UC3);

}

Point2f get_velocity(Point2f bl, Point2f ur){
    // findLK(Mat image);
    // int count = 0;
    // for(int i=0; i<points[1].size; i++){
    //     if(withinRect(points[1][i], bl, ur))
    //         count ++:
    // }
  cout<<bl<<endl;
    Mat flow, downsize;  
    // CvRect setRect = cvRect(100, 200, 200, 300); // ROI in image
    // cvSetImageROI(previous, setRect);
    cvtColor(image, downsize, COLOR_BGR2GRAY);
    if(previous.empty()){
      downsize.copyTo(previous);
    }

    Mat previous_roi(previous,Rect(bl,ur));
    Mat downsize_roi(downsize,Rect(bl,ur));
    calcOpticalFlowFarneback(previous_roi, downsize_roi, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    cv::Mat xy[2]; //X,Y
    cv::split(flow, xy);
    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);
    //cout<<magnitude;
    double mag_max;
    magnitude.convertTo(magnitude, CV_8UC1);
    angle.convertTo(angle, CV_8UC1);
    cv::minMaxLoc(magnitude, 0, &mag_max,0,0);
    //magnitude.convertTo(magnitude, -1, 255.0/mag_max);
    //cout<<mag_max<<endl;
    threshold(magnitude, magnitude, of_threshold, 1.0, CV_THRESH_BINARY);
    
    Mat mag_chans[4]; 
    split(magnitude,mag_chans);
    cout<<"mag mean "<<mean(mag_chans[0])[0]<<endl;
    Mat ang_chans[4]; 
    split(angle,ang_chans);
    //cout<<"angle mean "<<mean(ang_chans[0])<<endl;

    Mat average = mag_chans[0].mul(ang_chans[0]);
    //cout<<"angle real average "<<sum(average)/sum(mag_chans[0])<<endl;
    //magnitude.copyTo(flowImage);
    //draw optical flow 
    //colored_of = Mat(flow.rows,flow.cols, CV_8UC3);
    
    Point2f res(mean(mag_chans[0])[0],sum(average)[0]/sum(mag_chans[0])[0]);
    return res;
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
    if(stop_left)
      glColor4f(1.0, 1.0, 1.0,0.5); 
    glBegin(GL_POLYGON); 
    glVertex3f(0.0, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, BLOCK_SIZE); 
    glVertex3f(0, 0.0, BLOCK_SIZE); 
    glEnd(); 
    
    //right
    glTranslatef(2*BLOCK_SIZE, 0, 0);
    glColor4f(204.0/255.0, 1.0, 153.0/255.0,0.5);
    if(stop_right)
      glColor4f(1.0, 1.0, 1.0,0.5);  
    glBegin(GL_POLYGON); 
    glVertex3f(0.0, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, BLOCK_SIZE); 
    glVertex3f(0, 0.0, BLOCK_SIZE); 
    glEnd(); 
    //up
    glTranslatef(-BLOCK_SIZE, 0, BLOCK_SIZE);
    glColor4f(1.0, 1.0, 153.0/255.0,0.5); 
    if(stop_back)
      glColor4f(1.0, 1.0, 1.0,0.5); 
    glBegin(GL_POLYGON); 
    glVertex3f(0.0, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, 0.0); 
    glVertex3f(BLOCK_SIZE, 0.0, BLOCK_SIZE); 
    glVertex3f(0, 0.0, BLOCK_SIZE); 
    glEnd();
    //down
    glTranslatef(0, 0, -2*BLOCK_SIZE);
    glColor4f(153.0/255.0, 204.0/255.0, 1.0,0.5); 
    if(stop_front)
      glColor4f(1.0, 1.0, 1.0,0.5); 
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
        Mat overlay_test;//=  Mat::zeros(image.size(), CV_8UC3);
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
    // if(show_of_flag){

    //     glDrawPixels( colored_of.size().width, colored_of.size().height, GL_BGR, GL_UNSIGNED_BYTE, colored_of.ptr() );  
    // }

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
        case 'f':
            if(needToInit)
                needToInit = false;
            else
                needToInit = true;
            break;
        case 'o':
            if(show_of_flag)
                show_of_flag = false;
            else
               show_of_flag = true;  
            break;
        default:
            break;
    }
}

void idle()
{
    stop_front = false;
    stop_right = false;
    stop_left = false;
    stop_back = false;
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

    findLK(image);
    if(show_of_flag){
        if(get_velocity(front_bl,front_ur).x>mag_threshold){
          stop_front = true;
          cout<<"front"<<endl;
        }
        if(get_velocity(back_bl,back_ur).x>mag_threshold){
          stop_back = true;
          cout<<"back"<<endl;
        }
        if(get_velocity(left_bl,left_ur).x>mag_threshold){
          stop_left = true;
          cout<<"left"<<endl;
        }
        if(get_velocity(right_bl,right_ur).x>mag_threshold){
          stop_right = true;
          cout<<"right"<<endl;
        }

    }
    cvtColor(image, previous, COLOR_BGR2GRAY);
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
    
    delete[] music_file_path;
    return 0;
}

