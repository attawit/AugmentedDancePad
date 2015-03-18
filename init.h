#include "background_segm.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
//#include "audio.h"

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#define NUM_CELLS 4
#define ARROW_SIDE 40.0
#define PATTERN_COL_RATIO 0.20
#define PATTERN_HIT_LINE_RATIO 0.75
#define PATTERN_HIT_BOUND 20.0
#define PATTERN_TIME_UNIT 0.33
#define TIME_PREPARATION 5

using namespace cv;
using namespace std;

vector< vector<int> > patterns;
vector<float> times;

// pattern display
Mat arrow_left, arrow_right, arrow_up, arrow_down, arrow_left_hit, arrow_right_hit, arrow_up_hit, arrow_down_hit;
Mat pattern_area, pattern_area_bg_color;
double start_line_padding = 80.0;
double finish_line_padding = 80.0;
double pattern_alpha = 0.3;

bool move_front = false;
bool move_back = false;
bool move_left = false;
bool move_right = false;
bool stop_front = false;
bool stop_back = false;
bool stop_left = false;
bool stop_right = false;

int total_hit = 0;
int max_combo = 0;

Scalar line_color(150, 150, 150);

chrono::time_point<chrono::system_clock> timer_start;
//float timer_start = 0.0;

void readPatterns(const char* dir, const char* file_name){
    char file[100];
    strcpy(file,dir); // copy string one into the result.
    strcat(file,"/");
    strcat(file,file_name);
    cout << "opening pattern file from: " << file << endl;
    fstream infile(file, ios_base::in);
    if (!infile) {
        cout << "No pattern file specified in the music path." << endl;
        exit(0);
    }
    string temp;
    while (getline(infile, temp))
    {
        istringstream buffer(temp);
        int number;
        float time;
        vector<int> pattern;
        for (int i = 0; i < NUM_CELLS; i++){
            buffer >> number;
            pattern.push_back(number);
            if (number == 1) {
                total_hit++;
            }
        }
        buffer >> time;
        times.push_back(time);
        patterns.push_back(pattern);
    }
    
    cout << "pattern: " << endl;
    for (int i = 0; i < patterns.size(); i++) {
        cout << patterns[i][0] << ", " << patterns[i][1] << ", " << patterns[i][2] << ", " << patterns[i][3] << endl;
        cout << times[i] << endl;
    }

    
}

void readMusic(const char* dir, char* music_file_path, const char* file_name){
    strcpy(music_file_path,dir); // copy string one into the result.
    strcat(music_file_path,"/");
    strcat(music_file_path,file_name);
//    music_file_path = string(dir, 0, 100);
//    music_file_path += "/"+file_name;
    cout << "opening music file from: " << music_file_path << endl;
}

void readArrows(Mat& arrow_up, Mat& arrow_down, Mat& arrow_left, Mat&arrow_right, Mat& arrow_up_hit, Mat& arrow_down_hit, Mat& arrow_left_hit, Mat&arrow_right_hit){
    Mat arrow_up_tmp, arrow_down_tmp, arrow_left_tmp, arrow_right_tmp,arrow_up_hit_tmp, arrow_down_hit_tmp, arrow_left_hit_tmp, arrow_right_hit_tmp;
    arrow_left_tmp = cvLoadImage("./image/arrow_left.png", CV_LOAD_IMAGE_COLOR);
    arrow_right_tmp = cvLoadImage("./image/arrow_right.png", CV_LOAD_IMAGE_COLOR);
    arrow_up_tmp = cvLoadImage("./image/arrow_up.png", CV_LOAD_IMAGE_COLOR);
    arrow_down_tmp = cvLoadImage("./image/arrow_down.png", CV_LOAD_IMAGE_COLOR);
    arrow_left_hit_tmp = cvLoadImage("./image/arrow_left_hit.png", CV_LOAD_IMAGE_COLOR);
    arrow_right_hit_tmp = cvLoadImage("./image/arrow_right_hit.png", CV_LOAD_IMAGE_COLOR);
    arrow_up_hit_tmp = cvLoadImage("./image/arrow_up_hit.png", CV_LOAD_IMAGE_COLOR);
    arrow_down_hit_tmp = cvLoadImage("./image/arrow_down_hit.png", CV_LOAD_IMAGE_COLOR);
    resize(arrow_left_tmp, arrow_left, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_right_tmp, arrow_right, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_up_tmp, arrow_up, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_down_tmp, arrow_down, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_left_hit_tmp, arrow_left_hit, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_right_hit_tmp, arrow_right_hit, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_up_hit_tmp, arrow_up_hit, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_down_hit_tmp, arrow_down_hit, Size(ARROW_SIDE, ARROW_SIDE));
    
    flip(arrow_up, arrow_up, 0);
    flip(arrow_down, arrow_down, 0);
    flip(arrow_up_hit, arrow_up_hit, 0);
    flip(arrow_down_hit, arrow_down_hit, 0);
}

void readParameters(const char* fileName, Mat& cm, Mat& dc){
    ifstream inFile;
    inFile.open(fileName);
    if(!inFile.is_open()){
        cout<<"Could not open camera.txt"<<endl;
    }
    
    char buffer[30];
    inFile.getline(buffer,30);
    
    inFile >> cm.at<double>(0,0);
    inFile >>  cm.at<double>(0,1);
    inFile >>  cm.at<double>(0,2);
    inFile >>  cm.at<double>(1,0);
    inFile >>  cm.at<double>(1,1);
    inFile >>  cm.at<double>(1,2);
    inFile >>  cm.at<double>(2,0);
    inFile >>  cm.at<double>(2,1);
    inFile >>  cm.at<double>(2,2);
    char buffer2[30];
    inFile.getline(buffer2,30);
    inFile.getline(buffer2,30);
    inFile.getline(buffer2,30);
    inFile >>  dc.at<double>(0,0);
    inFile >>  dc.at<double>(1,0);
    inFile >>  dc.at<double>(2,0);
    inFile >>  dc.at<double>(3,0);
    inFile >>  dc.at<double>(4,0);
}
