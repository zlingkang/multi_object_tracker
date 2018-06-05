#include <iostream>
#include "det_and_track.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

DetAndTrack::DetAndTrack():
    get_new_detection_(false),
    first_time_detection_(true)
{
    face_cascade_.load("../haarcascade_frontalface_alt.xml");
    detection_sleep_time_ = 200;
    track_sleep_time_ = 30;
}

DetAndTrack::DetAndTrack(int _detection_sleep_time, int _track_sleep_time):
    get_new_detection_(false),
    first_time_detection_(true),
    detection_sleep_time_(_detection_sleep_time),
    track_sleep_time_(_track_sleep_time)
{
    face_cascade_.load("../haarcascade_frontalface_alt.xml");
}

void DetAndTrack::detectionTask()
{
    while(1)
    {
        std::cout << "detecting.." << std::endl;
        mutex_.lock();
        cv::Mat local_frame = current_frame_.clone();
        mutex_.unlock();

        std::vector<cv::Rect> local_boxes;
        face_cascade_.detectMultiScale(local_frame, local_boxes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30)); 
        std::this_thread::sleep_for(std::chrono::milliseconds(detection_sleep_time_));
        
        std::lock_guard<std::mutex> lockGuard(mutex_);
        det_boxes_ = local_boxes;
        //final_boxes_ = local_boxes;
        get_new_detection_ = true;
        std::cout << "detecting done" << std::endl;
    }
}

void DetAndTrack::trackTask()
{
    while(1)
    {
        std::cout << "tracking..." << std::endl;

        mutex_.lock();
        cv::Mat local_frame = current_frame_.clone();
        mutex_.unlock();

        if(get_new_detection_)
        {
            mutex_.lock();
            std::vector<cv::Rect> local_det_boxes = det_boxes_;
            mutex_.unlock();

            if(first_time_detection_)
            {
                track_manager_ptr_ = new TrackerManager(local_frame, local_det_boxes); 
                first_time_detection_ = false;
                std::cout << "track manager initialized" << std::endl;
            }
            else
            {
                std::cout << "track start update with new detection..." << std::endl;
                track_manager_ptr_ -> updateTrackersWithNewDetectionResults(local_det_boxes); 
            }

            std::lock_guard<std::mutex> lockguard(mutex_);
            get_new_detection_ = false;
        }
        else
        {
            if(!first_time_detection_)
            {
                std::cout << "track start update with new frame..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(track_sleep_time_));
                track_manager_ptr_->updateTrackersWithNewFrame(local_frame);
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }

        std::cout << "tracking done" << std::endl;
    }
}

void DetAndTrack::setFrame(const cv::Mat& _new_frame)
{
    current_frame_ = _new_frame.clone(); 
}
