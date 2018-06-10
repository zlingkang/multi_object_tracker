# multi_object_tracker
An optical flow and Kalman Filter based multi-ojbect tracker

# Introduction
Object detection is slow, especially for embedded platforms. Tracking algorithm implementations in OpenCV3 contrib does not work well
 for multi-object tracking, the processing time increases linearly with the number of trackers. And they are all long-term tracking 
 oriented. When we have detection every once a while, we do not need the trackers to be that accurate, and we need high speed tracking.  
 So I implemented this optical-flow and kalman filter based multi-object tracker, ~50ms processing time for a 640x480 frame tested on Odroid XU4.

# Description
* `lk_tracker.cpp` contains the tracker implementation, including `LkTracker` as an individual tracker and `TrackerManager` which manages all trackers
* `object_detection.cpp` contains an OpenCV face detector, you can modify the class to use other object detection. 
* `lk_tracker_test.cpp` contains a simple test in a single-threaded setting. It requires opencv3 contrib to draw the initial bounding box. 
Modify that if you do not have OpenCV 3 contrib. Or simply commment it out in CMakeLists.txt if you do not want it.
* `det_and_track.cpp` contatins the setting to use our tracker and a face detector in a multi-threaded setting. The face detector here can be
replaced with any detector you like.
* `main.cpp` contains the usage of the detection and tracking in multi-thread setting.

# Usage
* Modify the OpenCV path in CMakeLists.txt.
* Modify the detection_sleep_time_ and track_sleep_time_ to 0 when using with embedded platforms.  
``mkdir build``  
``cd build``  
``cmake ..``  
``make``  

# Tune Parameters  
* Use Kalman Filter or not:  
Modify the `USE_KF_` in TrackerManger's constructor in `lk_tracker.cpp`. If true, the Kalman Filter will be applied, and the trackers will be more robust against oclussion, however it may also cause drifting problem.  
* Matching cost threshold:  
`COST_THRESHOLD_` in `lk_tracker.cpp`, everytime we get detection results from the detector, we will match all the bouding boxes with the existing trackers using Hungarian algorithm. However, sometimes even when a detection result is matched with an existing trackers, it may still not be ideal. Thus we use the `COST_THRESHOLD_` to get rid of this match.
* MIN_ACCEPT_FRAMES_ and MIN_REJECT_FRAMES_ in lk_tracker.cpp  
If in continuous MIN_ACCPET_FRMAES_ frames, a tracker can be matched with a detection result, then this tracker is 'qualified' as a tracker to a most-likely real face. If in continuous MIN_REJECT_FRAMES_ frames, a tracker cannot be matched with detection result, then this tracker is most-likely lost tracking and will be removed.   
* Detection and tracking frequency:  
The `detection_sleep_time_` and `track_sleep_time_` together with the performanmce of you computer determines the detection and tracking frequency. Let's say it takes 100ms for detection for each frame, and you set detection_sleep_time_ to 150ms, then you will get detection results every 250ms. The smaller the value means the better results you can get but more load on your CPU. Usually if you want to get the best performance in embedded environment (Raspberry Pi or Odroid), just set them to 0.

# Related  
Checkout [https://github.com/zlingkang/mtcnn_face_detection_and_tracking/tree/master](https://github.com/zlingkang/mtcnn_face_detection_and_tracking/tree/master) to see how to replace the OpenCV face detector used in this repo to a deep learning based face detector which is much more accurate.

# Credits  
* OpenCV LK optical flow  
* Munkres (Hungarian) algorithm implementation: [https://github.com/soimy/munkres-opencv](https://github.com/soimy/munkres-opencv)
