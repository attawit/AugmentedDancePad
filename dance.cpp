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
#define BLOCK_SIZE 55
#define VELOCITY_THRESHOLD 15
#define HAS_OBJ_RATIO_THRESHOLD 0.01
#define FEATURENUM 80
#define VELOCITY_THRESHOLD_DIFF 10
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
bool needToInit_pre = false;
//ROI of the dance pad's blocks
Point2f front_bl, front_ur, left_bl, left_ur, right_bl, right_ur, back_bl, back_ur, middle_bl, middle_ur;

//direction in each block
Point2f front_v, left_v, right_v, back_v, middle_v;
int front_count, left_count, right_count, back_count, middle_count;

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
    std::this_thread::sleep_for (std::chrono::seconds(TIME_PREPARATION));
    std::cout << "pause of " << TIME_PREPARATION << " seconds ended\n";
    system("afplay ./music/jmww/music.wav");
}

bool withinRect(Point2f p, Point2f bl, Point2f ur){
    if(p.x>bl.x&&p.x<ur.x&&p.y>bl.y&&p.y<ur.y)
        return true;
    else
        return false;
}

//clear outliers of feature points
void clearOutlier(Point2f bl, Point2f ur){
  Point2f new_ur = Point2f(ur.x,ur.y+30);
  for(int i=0; i<points[1].size(); i++){
    if(!withinRect(points[1][i], bl, new_ur)){
      points[1].erase(points[1].begin()+i);
      points[0].erase(points[0].begin()+i);
      i--; 
    }
  }
}

void regenerate_features(Point2f bl, Point2f ur){
  int count = 0;
  for(int i=0; i<points[1].size(); i++){
      if(withinRect(points[1][i], bl, ur)){
        count++;          
      }    
  }
  if(count<FEATURENUM){
    needToInit = true;
    //cout<<"regenerate"<<endl;
  } 

}


void get_velocity(void){//use the result of LK
  front_v = Point2f(.0f,.0f);
  back_v = Point2f(.0f,.0f);
  left_v = Point2f(.0f,.0f);
  right_v = Point2f(.0f,.0f);
  middle_v = Point2f(.0f,.0f);

  front_count = 0;
  left_count = 0;
  right_count = 0;
  back_count = 0;
  middle_count = 0;
  //back_count2 = 0;
  Point2f new_front_ur = Point2f(front_ur.x,front_ur.y-30.0f);
  Point2f new_left_ur = Point2f(left_ur.x-20.0f,left_ur.y);
  Point2f new_right_bl = Point2f(right_bl.x+20.0f, right_bl.y);
  Point2f new_back_bl = Point2f(back_bl.x,back_bl.y);
  middle_bl = Point2f(left_ur.x,front_ur.y);
  middle_ur = Point2f(right_bl.x,right_ur.y);
  for(int i = 0; i<points[0].size(); i++){
    if(withinRect(points[0][i],front_bl,new_front_ur)){
      float x = points[0][i].x - points[1][i].x;
      float y = points[0][i].y - points[1][i].y;
      front_v.x -= x;
      front_v.y -= y;
      front_count ++;
    }else if(withinRect(points[0][i],new_back_bl,back_ur)){
      float x = points[0][i].x - points[1][i].x;
      float y = points[0][i].y - points[1][i].y;
      back_v.x -= x;
      back_v.y -= y;
      back_count++;
      //if(y)
    }else if(withinRect(points[0][i],left_bl,new_left_ur)){
      float x = points[0][i].x - points[1][i].x;
      float y = points[0][i].y - points[1][i].y;
      left_v.x -= x;
      left_v.y -= y;
      left_count ++;
    }else if(withinRect(points[0][i],new_right_bl,right_ur)){
      float x = points[0][i].x - points[1][i].x;
      float y = points[0][i].y - points[1][i].y;
      right_v.x -= x;
      right_v.y -= y;
      right_count ++;
    }else if(withinRect(points[0][i],middle_bl,middle_ur)){
      float x = points[0][i].x - points[1][i].x;
      float y = points[0][i].y - points[1][i].y;
      middle_v.x -= x;
      middle_v.y -= y;
      middle_count ++;
    }

  }

  if(front_count!=0){
    front_v.x /= front_count;
    front_v.y /= front_count;
  }
  if(back_count!=0){
    back_v.x /= back_count;
    back_v.y /= back_count;
    cout<<"back_y"<<back_v.y<<endl;
  }
  if(left_count!=0){
    left_v.x /= left_count;
    left_v.y /= left_count;
  }
  if(right_count!=0){
    right_v.x /= right_count;
    right_v.y /= right_count;
  }
  if(middle_count!=0){
    middle_v.x /= middle_count;
    middle_v.y /= middle_count;
  }
  //cout<<"fv"<<front_v<<endl;

}

Point2f get_velocity(Point2f bl, Point2f ur){//use the result of LK
  Point2f res = Point2f(.0f,.0f);
  for(int i = 0; i<points[0].size(); i++){
    if(withinRect(points[0][i],bl,ur)){
      float x = points[0][i].x - points[1][i].x;
      float y = points[0][i].y - points[1][i].y;
      res.x -= x;
      res.y -= y;
    }

  }

  cout<<"res "<<res<<endl;
  return res;
}

Point2f get_velocity2(Point2f bl, Point2f ur){//use the result of farne OF
    // findLK(Mat image);
    // int count = 0;
    // for(int i=0; i<points[1].size; i++){
    //     if(withinRect(points[1][i], bl, ur))
    //         count ++:
    // }
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
    //cout<<"mag mean "<<mean(mag_chans[0])[0]<<endl;
    Mat ang_chans[4]; 
    split(angle,ang_chans);
    //cout<<"angle mean "<<mean(ang_chans[0])<<endl;

    Mat average = mag_chans[0].mul(ang_chans[0]);
    //cout<<"angle real average "<<sum(average)[0]/sum(mag_chans[0])[0]<<endl;
    //magnitude.copyTo(flowImage);
    //draw optical flow 
    //colored_of = Mat(flow.rows,flow.cols, CV_8UC3);
    
    Point2f res(mean(mag_chans[0])[0],sum(average)[0]/sum(mag_chans[0])[0]);
    return res;
}

void findLK(Mat image){
    //LK optical flow
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    if(needToInit){

         // automatic initialization
        Rect userROI(Point2f(front_bl.x,front_ur.y),Point2f(right_bl.x,back_bl.y));
        //cout<<"ROI"<<userROI<<endl;
        Mat mask = Mat::zeros(gray.size(), CV_8UC1);  // type of mask is CV_8U
        mask(userROI) = 1; 
       //   cout<<"generate point single passenger"<<endl;
        goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, mask, 3, 0, 0.04);
            //cout<<points[1]<<endl;
        cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            //addRemovePt = false;
    }else if(!points[0].empty()){
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
        clearOutlier(Point2f(left_bl.x,front_bl.y),Point2f(right_ur.x+50,back_ur.y));
        get_velocity();
    }
    
    needToInit = false;
    if(!needToInit&&!points[0].empty()&&!points[1].empty()){
      
      regenerate_features(Point2f(left_bl.x,front_bl.y),Point2f(right_ur.x+50,back_ur.y));

    }
    std::swap(points[1], points[0]);
    cv::swap(prevGray, gray);

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
    glColor4f(153.0/255.0, 204.0/255.0, 1.0,0.5); 
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
    glColor4f(1.0, 1.0, 153.0/255.0,0.5); 
    if(stop_front){
      glColor4f(1.0, 1.0, 1.0,0.5); 
      //cout<<"change color"<<endl;
    }
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
    back_ur.x = tx+BLOCK_SIZE*0.3;
    gluProject(squareSize*10+BLOCK_SIZE*2, 0, -squareSize*2+BLOCK_SIZE*2, _modelview, _projection, _viewport, &tx, &ty, &tz);
    back_ur.y = ty;

}

void drawDancePad2()
{
    glDisable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);
    GLUquadricObj *qobj = gluNewQuadric();
    glPushMatrix();

    //left   
    glTranslatef(squareSize*10, 0, -squareSize*2);
    glColor3f(1.0, 153.0/255.0, 153.0/255.0); 
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
    glColor3f(204.0/255.0, 1.0, 153.0/255.0);
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
    glColor3f(153.0/255.0, 204.0/255.0, 1.0); 
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
    glColor3f(1.0, 1.0, 153.0/255.0); 
    if(stop_front){
      glColor4f(1.0, 1.0, 1.0,0.5); 
      //cout<<"change color"<<endl;
    }
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
    back_ur.x = tx+BLOCK_SIZE*0.3;
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


bool hasObjInRoi(Point2f rect_bl, Point2f rect_ur){
  //cout<<rect_bl<<" "<<rect_ur<<endl;
    Mat mat(background_image, Rect(rect_bl, rect_ur));
    // count how many pixels is filled
    int count = countNonZero(mat);
    //cout<<"nonzero in bs "<<count<<endl;
    // if the count is greater than some ratio of the whole roi,
    // we say it contains object
    if (count > (rect_ur.x-rect_bl.x)*(rect_ur.y-rect_bl.y) * HAS_OBJ_RATIO_THRESHOLD){
      //cout<<"has obj"<<endl;
      return true;
    }else{
      return false;
    }
        
}

void motionDetectionHelper(int id, bool& pre, bool& stop, Point2f v, Point2f rect_bl, Point2f rect_ur){
    //Point2f velocity = get_velocity(rect_bl, rect_ur);
    Point2f velocity = v;
    //cout<<"v "<<v<<endl;
    if (pre) {
        if (!hasObjInRoi(rect_bl, rect_ur)) {
            // if there is no object in the roi, set pre_fron to false for the next frame
            pre = false;
        }else{
            //if (velocity.y < 0) {
                if (get_vel_length(velocity) > VELOCITY_THRESHOLD) {
                    // if it's moving, set pre_front for the next frame
                    pre = true;
                }else{
                    pre = false;
                    stop = true;
                    //cout<<"HIT!"<<endl;
                }
            // }else{
            //     pre = false;
            // }
        }
    }else{
        // if the velocity in the y axis is equal to or greater than 0, than the sub-velocity is to the upside, just discard it.
        //if (velocity.y < 0) {
      if(id==2){
        if(velocity.y > 0 && (get_vel_length(velocity)-get_vel_length(middle_v)>VELOCITY_THRESHOLD_DIFF)){
         if (get_vel_length(velocity) > VELOCITY_THRESHOLD) {
                // if it's moving, set pre_front for the next frame
                pre = true;
            }
        }
      }else if(id==1){
        if(velocity.y < 0){
         if (get_vel_length(velocity) > VELOCITY_THRESHOLD) {
                // if it's moving, set pre_front for the next frame
                pre = true;
            }
        }
      }else{
        if (get_vel_length(velocity) > VELOCITY_THRESHOLD) {
            // if it's moving, set pre_front for the next frame
            pre = true;
        }
      }
    }
}

void motionDetection(){
    // detect whether there is movement following a stop in each roi

    // motionDetectionHelper(move_front, stop_front, front_bl, front_ur);
    // motionDetectionHelper(move_back, stop_back, back_bl, back_ur);
    // motionDetectionHelper(move_left, stop_left, left_bl, left_ur);
    // motionDetectionHelper(move_front, stop_front, front_bl, front_ur);
    
    motionDetectionHelper(1,move_front, stop_front, front_v, front_bl, front_ur);
    motionDetectionHelper(2,move_back, stop_back, back_v, back_bl, back_ur);
    motionDetectionHelper(3,move_left, stop_left, left_v, left_bl, left_ur);
    motionDetectionHelper(4,move_right, stop_right, right_v, right_bl, right_ur);

}

void display()
{
    // clear the window
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (image.channels() == 3) {
//        pattern_area = image(Rect(0, 0, image.cols*PATTERN_COL_RATIO, image.rows));
//        pattern_area_bg_color =  Mat(Size(width*PATTERN_COL_RATIO, height), CV_8UC3, cv::Scalar(255, 255, 255));
//        cv::addWeighted(pattern_area_bg_color, pattern_alpha, pattern_area, 1.0 - pattern_alpha , 0.0, pattern_area);
        
//        line(image, Point(0, 0), Point(0, image.rows), Scalar(255, 0, 0));
//        line(image, Point(PATTERN_COL_RATIO*image.cols, 0), Point(PATTERN_COL_RATIO*image.cols, image.rows), Scalar(255, 0, 0));
//        for (int i = 1; i <= NUM_CELLS ; i++) {
//            line(image, Point(PATTERN_COL_RATIO*image.cols*i/(NUM_CELLS+1), 0), Point(PATTERN_COL_RATIO*image.cols*i/(NUM_CELLS+1), image.rows), line_color);
//        }
        
        Mat left_area = image(Rect(0, 0, image.cols*PATTERN_COL_RATIO/4, image.rows));
        Mat up_area = image(Rect(image.cols*PATTERN_COL_RATIO*1/4, 0, image.cols*PATTERN_COL_RATIO/4, image.rows));
        Mat down_area = image(Rect(image.cols*PATTERN_COL_RATIO*2/4, 0, image.cols*PATTERN_COL_RATIO/4, image.rows));
        Mat right_area = image(Rect(image.cols*PATTERN_COL_RATIO*3/4, 0, image.cols*PATTERN_COL_RATIO/4, image.rows));
        
        Mat left_area_color =  Mat(left_area.size(), CV_8UC3, cv::Scalar(153, 153, 255));
        Mat up_area_color =  Mat(left_area.size(), CV_8UC3, cv::Scalar(153, 255, 255));
        Mat down_area_color =  Mat(left_area.size(), CV_8UC3, cv::Scalar(255, 153, 153));
        Mat right_area_color =  Mat(left_area.size(), CV_8UC3, cv::Scalar(153, 255, 204));
        
        cv::addWeighted(left_area_color, pattern_alpha, left_area, 1.0 - pattern_alpha , 0.0, left_area);
        cv::addWeighted(up_area_color, pattern_alpha, up_area, 1.0 - pattern_alpha , 0.0, up_area);
        cv::addWeighted(down_area_color, pattern_alpha, down_area, 1.0 - pattern_alpha , 0.0, down_area);
        cv::addWeighted(right_area_color, pattern_alpha, right_area, 1.0 - pattern_alpha , 0.0, right_area);
        
        
        line(image, Point(0, start_line_padding), Point(PATTERN_COL_RATIO*image.cols, start_line_padding), Scalar(60, 60, 60), 3);
        line(image, Point(0, image.rows-finish_line_padding), Point(PATTERN_COL_RATIO*image.cols, image.rows-finish_line_padding), Scalar(60, 60, 60), 3);
        
        Mat hit_area = image(Rect(0, image.rows*PATTERN_HIT_LINE_RATIO-PATTERN_HIT_BOUND, image.cols*PATTERN_COL_RATIO, 2*PATTERN_HIT_BOUND));
        Mat hit_area_color =  Mat(hit_area.size(), CV_8UC3, cv::Scalar(60, 60, 60));
        cv::addWeighted(hit_area_color, 0.3, hit_area, 0.7 , 0.0, hit_area);
//        line(image, Point(0, image.rows*PATTERN_HIT_LINE_RATIO), Point(PATTERN_COL_RATIO*image.cols, image.rows*PATTERN_HIT_LINE_RATIO), line_color);
        line(image, Point(0, image.rows*PATTERN_HIT_LINE_RATIO-PATTERN_HIT_BOUND), Point(PATTERN_COL_RATIO*image.cols, image.rows*PATTERN_HIT_LINE_RATIO-PATTERN_HIT_BOUND), line_color);
        line(image, Point(0, image.rows*PATTERN_HIT_LINE_RATIO+PATTERN_HIT_BOUND), Point(PATTERN_COL_RATIO*image.cols, image.rows*PATTERN_HIT_LINE_RATIO+PATTERN_HIT_BOUND), line_color);
    }
    
    
    // if(background_image_flag){
    //     cvtColor( background_image, image, CV_GRAY2BGR );
    //     //flip(image,image, -1);
    // }
        //flip(background_image, image, 0);
    
    
    glDisable(GL_DEPTH_TEST);
    //glDrawPixels( image.size().width, image.size().height, GL_BGR, GL_UNSIGNED_BYTE, image.ptr() );
    //alpha blend test start
    if(background_image_flag){
    //if(plane_detection_flag && !image.empty()){    
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
        Point2f new_front_ur = Point2f(front_ur.x,front_ur.y-40.0f);
        Point2f new_left_ur = Point2f(left_ur.x-20.0f,left_ur.y);
        Point2f new_right_bl = Point2f(right_bl.x+20.0f, right_bl.y);
        Point2f new_back_bl = Point2f(back_bl.x,back_bl.y-15.0f);
        //rectangle(overlay_test, left_bl, new_left_ur,cvScalar(0, 255, 255, 0), CV_FILLED, 8, 0);
        //rectangle(overlay_test, new_right_bl, right_ur,cvScalar(0, 255, 255, 0), CV_FILLED, 8, 0);

          //Point2f new_front_ur = Point2f(front_ur.x,front_ur.y-50.0f);
        //rectangle(overlay_test, front_bl, new_front_ur,cvScalar(0, 255, 255, 0), CV_FILLED, 8, 0);
        //rectangle(overlay_test, new_back_bl, back_ur,cvScalar(0, 255, 255, 0), CV_FILLED, 8, 0);
       
        glDrawPixels( overlay_test.size().width, overlay_test.size().height, GL_BGRA, GL_UNSIGNED_BYTE, overlay_test.ptr() );
        
    }else{
        glDrawPixels( image.size().width, image.size().height, GL_BGR, GL_UNSIGNED_BYTE, image.ptr() );
    }
    //alpha blend test end
    glEnable(GL_DEPTH_TEST);

    
    //////////////////////////////////////////////////////////////////////////////////
    // Here, set up new parameters to render a scene viewed from the camera.
    
    //set viewport
    glViewport(0, 0, image.size().width, image.size().height);
    
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
    

    
    //glDrawPixels( image.size().width, image.size().height, GL_BGR, GL_UNSIGNED_BYTE, image.ptr() );
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
   
    }/*else{
        
        // glDrawPixels( image.size().width, image.size().height, GL_BGR, GL_UNSIGNED_BYTE, image.ptr() );
        // if(plane_detection_flag&&found){
        //   glPushMatrix();
        //   //move to the position where you want the 3D object to go
        //   glTranslatef(0, 0, 0); //this is an arbitrary position for demonstration
        //   //you will need to adjust your transformations to match the positions where
        //   //you want to draw your objects(i.e. chessboard center, chessboard corners)
        //   glRotatef(90, -1.0, 0.0, 0.0);
          
        //   drawAxes(squareSize);
        //   //showCorners();
        //   drawDancePad2();

        //   glPopMatrix();
        //   cout<<"wow"<<endl;
        // }
    }*/
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
//    clock_t t = clock();
    switch ( key )
    {
        case 'q':
            // quit when q is pressed
            exit(0);
            break;
        case 's':
            start_play = !start_play;
            // adjust the the base line of the timer
            //timer_start = float(t)/CLOCKS_PER_SEC;
            timer_start = chrono::system_clock::now();
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
        case '1':
            stop_left = true;
            break;
        case '2':
            stop_front = true;
            break;
        case '3':
            stop_back = true;
            break;
        case '4':
            stop_right = true;
            break;
        case '0':
            cout << "oh, hit!" << endl;
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

    front_v = Point2f(.0f,.0f);
    back_v = Point2f(.0f,.0f);
    left_v = Point2f(.0f,.0f);
    right_v = Point2f(.0f,.0f);

    // grab a frame from the camera
    (*cap) >> image;
    flip(image, image, 1);

    //count down
    if(start_play&&timer()<=0){
      CvFont font;
      cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 10, 10, 10, 8, 8);
      IplImage* ipltemp;
      ipltemp = cvCreateImage(cvSize(image.cols, image.rows),8,3);
      IplImage iplt = image;
      cvCopy(&iplt, ipltemp);
      std::string s = std::to_string((int)ceil(abs(timer())));
      char const *pchar = s.c_str();  //use char const* as target type
      cvPutText(ipltemp, pchar, cvPoint(width/2-80,height/2+100), &font, cvScalar(255,255,255));
      image = cv::Mat(ipltemp);
    }

    flip(image, image, 0);

    if(start_play){
      plane_detection_flag = false;
      if(timer()>0){
        findLK(image);
        motionDetection();
        background_image_flag = true;
      }
      start_pattern(image);
    }    

    if (image.empty()) {
        cout << "No captured frame" << endl;
        exit(0);
    }
    if(start_play){
      background_subtraction();
    }
    
    if(plane_detection_flag){
       plane_detection(); 
    }
    
    if(start_play&&timer()>0&&!needToInit_pre) {
      needToInit = true;
      needToInit_pre = true;
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
    readArrows(arrow_up, arrow_down, arrow_left, arrow_right, arrow_up_hit, arrow_down_hit, arrow_left_hit, arrow_right_hit);
    
    //read music
    music_file_path = new char[100];
    readMusic(argv[2], music_file_path, MUSIC_FILE);
    
    //pattern display area
    
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

