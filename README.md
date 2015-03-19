# AugmentedDancePad

Augmented Dance Pad is an application that provides the music that users can dance to and captures the gait movement of the users dancing on an augmented dance pad projected to the ground. 

For technical details and other resources, please visit our website: https://sites.google.com/site/ardancepad/

To run the project, you will need to clone the repo, get Makefile with cmake and compile it:

git clone --recursive https://github.com/gmy/AugmentedDancePad <br \>
cd AugmentedDancePad <br \>
cmake . <br \>
make <br \>
./dance [camera_calibration_file] [path_of_your_pattern_and_music_folder] <br \>
(e.g. ./dance calibration_result_my.txt music/jmww)

Music should be in .wav format. <br \>
It's required to use name "pattern,txt" for the pattern file and "music.wav" for music file.


