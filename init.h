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
#define PATTERN_COL_RATIO 0.25
#define PATTERN_TIME_UNIT 0.15

using namespace cv;
using namespace std;

vector< vector<int> > patterns;

// pattern display
Mat arrow_left, arrow_right, arrow_up, arrow_down;
double start_line_padding = 100.0;
double finish_line_padding = 50.0;

float timer_start = 0.0;

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

void readMusic(const char* dir, char* music_file_path, const char* file_name){
    strcpy(music_file_path,dir); // copy string one into the result.
    strcat(music_file_path,"/");
    strcat(music_file_path,file_name);
//    music_file_path = string(dir, 0, 100);
//    music_file_path += "/"+file_name;
    cout << "opening music file from: " << music_file_path << endl;
}

void readArrows(Mat& arrow_up, Mat& arrow_down, Mat& arrow_left, Mat&arrow_right){
    Mat arrow_up_tmp, arrow_down_tmp, arrow_left_tmp, arrow_right_tmp;
    arrow_left_tmp = cvLoadImage("./image/arrow_left.png", CV_LOAD_IMAGE_COLOR);
    arrow_right_tmp = cvLoadImage("./image/arrow_right.png", CV_LOAD_IMAGE_COLOR);
    arrow_up_tmp = cvLoadImage("./image/arrow_up.png", CV_LOAD_IMAGE_COLOR);
    arrow_down_tmp = cvLoadImage("./image/arrow_down.png", CV_LOAD_IMAGE_COLOR);
    resize(arrow_left_tmp, arrow_left, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_right_tmp, arrow_right, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_up_tmp, arrow_up, Size(ARROW_SIDE, ARROW_SIDE));
    resize(arrow_down_tmp, arrow_down, Size(ARROW_SIDE, ARROW_SIDE));
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
