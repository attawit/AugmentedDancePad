#include "init.h"

float squareSize = 5;


float timer(){
    clock_t t = clock();
    float now = float(t)/CLOCKS_PER_SEC;
    return now-timer_start;
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

// a useful function for displaying your coordinate system
void drawAxes(float length)
{
    glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT) ;
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) ;
    glDisable(GL_LIGHTING) ;
    glTranslatef(.0f,-squareSize,0.0f);
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
    //cout<<foreground.channels()<<endl;
    for (int i = y; i < y+height; i++){
        for(int j = x; j < x+width; j++){
            double opacity =((double)foreground.data[(i-y) * foreground.step + (j-x) * foreground.channels() + 3])/ 255.;
            //cout<<opacity<<endl;

            for(int c = 0; opacity < 0.999 && c < background.channels(); ++c)
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
        line(image, Point(PATTERN_COL_RATIO*image.cols*i/(NUM_CELLS+1), 0), Point(PATTERN_COL_RATIO*image.cols*i/(NUM_CELLS+1), image.rows), Scalar(255, 0, 0));
    }
    
    line(image, Point(0, start_line_padding), Point(PATTERN_COL_RATIO*image.cols, start_line_padding), Scalar(255, 0, 0));
    line(image, Point(0, image.rows-finish_line_padding), Point(PATTERN_COL_RATIO*image.cols, image.rows-finish_line_padding), Scalar(255, 0, 0));
    
    float time_now = timer();
    cout << "time now: " << time_now << endl;
    // for each line of the pattern
    for (int i = 0; i < patterns.size(); i++) {
        float distance = (time_now - i * PATTERN_TIME_UNIT) * (ARROW_SIDE*10);
        
        // if it's not the turn to show up for the rest, just break the loop
        if (distance < 0)
            break;
        
        // if it's already finished, just skip it to conitue
        if (distance > image.rows - start_line_padding - finish_line_padding) {
            continue;
        }
        
        if (patterns[i][0] == 1) {
            overlay(arrow_left, image, (int)(PATTERN_COL_RATIO*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
        if (patterns[i][1] == 1) {
            overlay(arrow_up, image, (int)(PATTERN_COL_RATIO*2*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
        if (patterns[i][2] == 1) {
            overlay(arrow_down, image, (int)(PATTERN_COL_RATIO*3*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
        if (patterns[i][3] == 1) {
            overlay(arrow_right, image, (int)(PATTERN_COL_RATIO*4*image.cols/(NUM_CELLS+1)-ARROW_SIDE/2), (int)(start_line_padding+(distance-ARROW_SIDE/2)), (int)ARROW_SIDE, (int)ARROW_SIDE);
        }
        
    }
}
