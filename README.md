# multi_object_tracker
An optical flow and Kalman Filter based multi-ojbect tracker

# Introduction
Object detection is slow, especially for embedded platforms. Tracking algorithm implementations in OpenCV3 contrib does not work well
 for multi-object tracking, the processing time increases linearly with the number of trackers. And they are all long-term tracking 
 oriented. When we have detection every once a while, we do not need the trackers to be that accurate, and we need high speed tracking.  
 So I implemented this optical-flow and kalman filter based multi-object tracker, ~50ms processing time for a 640x480 frame tested on Odroid XU4.

# Description
* The lk_tracker.cpp contains the tracker implementation. 
* lk_tracker_test.cpp contains a simple test in a single-threaded setting. It requires opencv3 contrib to draw the initial bounding box. 
Modify that if you do not have OpenCV 3 contrib. Or simply commment it out in CMakeLists.txt if you do not want it.
* det_and_track.cpp contatins the setting to use our tracker and a face detector in a multi-threaded setting. The face detector here can be
replaced with any detector you like.
* main.cpp contains the usage of the detection and tracking in multi-thread setting.

# Usage
* Modify the OpenCV path in CMakeLists.txt.
* Modify the detection_sleep_time_ and track_sleep_time_ to 0 when using with embedded platforms.  
``mkdir build``  
``cd build``  
``cmake ..``  
``make``  

# Credits  
* OpenCV LK optical flow  
* Munkres (Hungarian) algorithm implementation: (https://github.com/soimy/munkres-opencv)[https://github.com/soimy/munkres-opencv]
