#ifndef DET_AND_TRACK_H
#define DET_AND_TRACK_H

class TrackerManager;

#include <opencv2/core/utility.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp> 
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

#include "lk_tracker.h"
#include "object_detection.h" 

class DetAndTrack
{
    bool get_new_detection_;
    bool first_time_detection_;
    std::vector<cv::Rect> det_boxes_;
    //std::vector<cv::Rect> track_boxes_;
    //std::vector<cv::Rect> final_boxes_;

    cv::Mat current_frame_;
    //cv::Mat last_frame_;

    //cv::CascadeClassifier face_cascade_;
    ObjectDetection* object_detection_ptr_;

    TrackerManager* track_manager_ptr_; 

    int detection_sleep_time_; //milliseconds
    int track_sleep_time_;

    public:
        std::mutex mutex_;
        DetAndTrack();
        DetAndTrack(int _detection_sleep_time, int _track_sleep_time);
        void detectionTask();
        void trackTask();
        
        std::vector<cv::Rect> getBox()
        {
            if(!first_time_detection_) // means the track_manager_ptr_ is initialized
            {
                return track_manager_ptr_->getAllBox();
            }
            else
            {
                std::vector<cv::Rect> res;
                return res;
            }
        }

        std::vector<cv::Scalar> getColor()
        {
            if(!first_time_detection_)
            {
                return track_manager_ptr_->getAllColor();
            }
            else
            {
                std::vector<cv::Scalar> res;
                return res;
            }
        }

        void setFrame(const cv::Mat& _new_frame);
};

#endif
