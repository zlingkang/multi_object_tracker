#include "lk_tracker.h"

LkTracker::LkTracker(const cv::Mat& _frame, const cv::Rect& _bbox, const int _tracker_id, const bool _use_kf):
    USE_KF_(_use_kf),
    tracker_id_(_tracker_id),
    bbox_(_bbox),
    status_(true),
    MIN_ACCEPT_FRAMES_(3),
    MIN_REJECT_FRAMES_(3),
    accepted_(false),
    rejected_(false),
    getting_frames_(0),
    missing_frames_(0),
    kf_(6, 4, 0),
    kf_state_(6, 1, CV_32F),
    kf_measure_(4, 1, CV_32F),
    first_time_(true)
{
    std::vector<cv::KeyPoint> kps;
    detector_->detect(_frame(_bbox), kps);
    cv::KeyPointsFilter::retainBest(kps, MAX_TRACK_POINTS_NUM_);
    for(auto kp:kps)
    {
        kp.pt.x += _bbox.x;
        kp.pt.y += _bbox.y;
        track_points_.push_back(kp.pt);
    }
    MIN_TRACK_POINTS_NUM_ = track_points_.size()*2/3;

    frame_width_ = _frame.cols;
    frame_height_ = _frame.rows;

    std::srand(std::time(0));
    color_ = cv::Scalar(std::rand()%255, std::rand()%255, std::rand()%255);

    // Kalman filter stuff

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf_.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf_.measurementMatrix = cv::Mat::zeros(4, 6, CV_32F);
    kf_.measurementMatrix.at<float>(0) = 1.0f;
    kf_.measurementMatrix.at<float>(7) = 1.0f;
    kf_.measurementMatrix.at<float>(16) = 1.0f;
    kf_.measurementMatrix.at<float>(23) = 1.0f;
   
    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf_.processNoiseCov.at<float>(0) = 0.05; //1e-2;
    kf_.processNoiseCov.at<float>(7) = 0.05; //1e-2;
    kf_.processNoiseCov.at<float>(14) = 0.05; //5.0f;
    kf_.processNoiseCov.at<float>(21) = 0.05; //5.0f;
    kf_.processNoiseCov.at<float>(28) = 0.05; //1e-2;
    kf_.processNoiseCov.at<float>(35) = 0.05; //1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(1e-1));
    
    ticks_ = cv::getTickCount();

}

void LkTracker::KalmanPredict()
{
    double prec_tick = ticks_;
    ticks_ = (double) cv::getTickCount();
    double dT = (ticks_ - prec_tick) / cv::getTickFrequency();
    kf_.transitionMatrix.at<float>(2) = dT;
    kf_.transitionMatrix.at<float>(9) = dT;
    kf_state_ = kf_.predict();
     
    bbox_.width = kf_state_.at<float>(4);
    bbox_.height = kf_state_.at<float>(5);
    bbox_.x = kf_state_.at<float>(0) - bbox_.width/2;
    bbox_.y = kf_state_.at<float>(1) - bbox_.height / 2;
    if(bbox_.x < 0)
    {
        bbox_.x = 0;
    }
    if(bbox_.y < 0)
    {
        bbox_.y = 0;
    }
    if(bbox_.width < 0)
    {
        bbox_.width = 1;
    }
    if(bbox_.height < 0)
    {
        bbox_.height = 1;
    }
    if((bbox_.y + bbox_.height >= frame_height_) || bbox_.y >= frame_height_)
    {
        if(bbox_.y >= frame_height_)
        {
            bbox_.y = frame_height_ - 2;
            bbox_.height = 1;
        }
        bbox_.height = frame_height_ - bbox_.y - 1;
    }
    if((bbox_.x + bbox_.width >= frame_width_) || bbox_.x >= frame_width_)
    {
        if(bbox_.x >= frame_width_)
        {
            bbox_.x = frame_width_ - 2;
            bbox_.width = 1;
        }
        bbox_.width = frame_width_ - bbox_.x - 1;
    }
    std::cout << "frame width frame height:" << frame_width_ << " " << frame_height_ << std::endl; 
}

void LkTracker::KalmanUpdate(cv::Rect _new_box)
{
    /*
    if(!accepted_)
    {
        found_frames_ ++;
        if(found_frames_ >= MIN_ACCEPT_FRAMES_)
        {
            accepted_ = true;
        }
    }
    missing_frames_ = 0;
*/
    kf_measure_.at<float>(0) = _new_box.x + _new_box.width/2;
    kf_measure_.at<float>(1) = _new_box.y + _new_box.height/2;
    kf_measure_.at<float>(2) = _new_box.width;
    kf_measure_.at<float>(3) = _new_box.height;

    if(first_time_)
    {
        first_time_ = false;
        kf_.errorCovPre.at<float>(0) = 1; // px
        kf_.errorCovPre.at<float>(7) = 1; // px
        kf_.errorCovPre.at<float>(14) = 1;
        kf_.errorCovPre.at<float>(21) = 1;
        kf_.errorCovPre.at<float>(28) = 1; // px
        kf_.errorCovPre.at<float>(35) = 1; // px

        kf_state_.at<float>(0) = kf_measure_.at<float>(0);
        kf_state_.at<float>(1) = kf_measure_.at<float>(1);
        kf_state_.at<float>(2) = 0;
        kf_state_.at<float>(3) = 0;
        kf_state_.at<float>(4) = kf_measure_.at<float>(2);
        kf_state_.at<float>(5) = kf_measure_.at<float>(3);
        // <<<< Initialization

        kf_.statePost = kf_state_;
    }
    else
    {
        kf_.correct(kf_measure_);
    }
}

template <class T>
T LkTracker::findMedian(std::vector<T> vec)
{
    size_t size = vec.size();

    if(size == 1)
    {
        return vec[0];
    }
    else
    {
        std::sort(vec.begin(), vec.end());
        if(size % 2 == 0)
        {
            return (vec[size/2 - 1] + vec[size/2]) / 2;
        }
        else
        {
            return vec[size/2];
        }
    }
}

void LkTracker::updateLkTracker(const cv::Mat& _frame)
{
    // Update points
    // by the end of this process, track_points_ contains all the currently tracked points in the current frame
    // track_points_old contains all the positions of the currently tracked points in the last frame
    // This part has been moved to trakcer_manager class update all points
    /*
    std::vector<cv::Point2f> next_keypoints;
    std::vector<cv::Point2f> prev_keypoints;
    for(auto kp:track_points_)
    {
        prev_keypoints.push_back(kp);
    }
    std::vector<unsigned char> status;
    std::vector<float> error;
     
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(_last_frame, _frame, prev_keypoints, next_keypoints, status, error);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    std::cout<<"Optical flow time: " << time_used.count() << "seconds" << std::endl;
  
    
    std::vector<int> vec_x;
    std::vector<int> vec_y;
    std::vector<cv::Point2f>old_track_points; 
    int i = 0;
    for(auto iter = track_points_.begin(); iter != track_points_.end(); i ++)
    {
        if(status[i] == 0)
        {
            iter = track_points_.erase(iter);
            continue;
        }
        vec_x.push_back(next_keypoints[i].x - iter->x);
        vec_y.push_back(next_keypoints[i].y - iter->y);
        old_track_points.push_back(*iter);
        *iter = next_keypoints[i];
        iter ++;
    }
    */
   
    std::cout << "start updating lktracker" <<std::endl;

    std::vector<int> vec_x;
    std::vector<int> vec_y;
    int ind = 0;
    std::cout << "old points: " << old_track_points_.size() << std::endl;
    std::cout << "new points: " << track_points_.size() << std::endl;
    for(auto iter = track_points_.begin(); iter != track_points_.end(); iter ++)
    {
        vec_x.push_back(iter->x - old_track_points_[ind].x);
        vec_y.push_back(iter->y - old_track_points_[ind].y);
        ind ++;
    }


    // Update bounding box
    std::cout << "  start updating bounding box" <<std::endl;
    int median_x = findMedian(vec_x);
    int median_y = findMedian(vec_y);
    std::vector<float> vec_scale;
    auto iter1 = track_points_.begin();
    
    for(size_t i = 0; i < old_track_points_.size(); i ++)
    {   
        iter1 ++;
        auto iter2 = iter1;
        iter1 --;
        for(size_t j = i+1; j < old_track_points_.size(); j ++)
        {
            auto distance = [](cv::Point2f pa, cv::Point2f pb)
            {
                return sqrt((pa.x-pb.x)*(pa.x-pb.x) + (pa.y-pb.y)*(pa.y-pb.y));
            };
            float old_dis = distance(old_track_points_[i], old_track_points_[j]);
            float new_dis = distance(*iter1, *iter2);
            if(old_dis > 0.001)
            {
                vec_scale.push_back(new_dis/old_dis);
            }
            iter2 ++;
        }
        iter1 ++;
    }
    
    float median_scale = findMedian(vec_scale) * 1.003;
    if(median_scale > SCALE_THRESHOLD)
    {
        median_scale = SCALE_THRESHOLD;
    }
    int new_width = bbox_.width * median_scale;
    int new_height = bbox_.height * median_scale;
    if(new_width < 10 || new_height < 10)
    {
        status_ = false;
        new_width = 10;
        new_height = 10;
    }
    int new_x = bbox_.x + median_x - (new_width - bbox_.width)/2;
    int new_y = bbox_.y + median_y - (new_height - bbox_.height)/2;
    if(new_x >= frame_width_ || new_y >= frame_height_)
    {
        status_ = false;
        throw std::invalid_argument("bounding box deprecated"); 
    }
    if(new_x < 0)
    {
        new_x = 0; 
    }
    if(new_x + new_width > frame_width_)
    {
        new_width = frame_width_ - new_x - 2;
    }
    if(new_y < 0)
    {
        new_y = 0;
    }
    if(new_y + new_height > frame_height_)
    {
        new_height = frame_height_ - new_y - 2;
    }
    cv::Rect new_box = cv::Rect(new_x, new_y, new_width, new_height);
    std::cout << "new_box" << new_box << std::endl; 
    
    bbox_ = new_box;
    
    if(USE_KF_)
    {
        KalmanUpdate(new_box);
    }
    else
    {
        bbox_ = new_box;
    }

    if(!first_time_)
    {
        if(USE_KF_)
        {
            KalmanPredict();
            std::cout << "kalman predict" << std::endl;
        }
    }
    std::cout << "bbox" << bbox_ << std::endl;

    // get rid of points out of the bounding box
    std::cout << "  start getting rid of points out of box" << std::endl;
    for(auto iter = track_points_.begin(); iter != track_points_.end(); )
    {
        auto inBox = [](cv::Point2f point, cv::Rect rec)
        {
            if(point.x > rec.x && point.x < (rec.x+rec.width) && point.y > rec.y && point.y < (rec.y+rec.height))
            {
                return true;
            }
            else
            {
                return false;
            }
        };
        if(!inBox(*iter, bbox_))
        {
            iter = track_points_.erase(iter);
            continue;
        }
        iter ++;
    }

    // get more keypoints in the bounding box if necessary
    std::cout << "  get more points in box" << std::endl << bbox_ <<std::endl;
    if(track_points_.size() < MIN_TRACK_POINTS_NUM_)
    {
        track_points_.clear();
        std::vector<cv::KeyPoint> kps;
        detector_->detect(_frame(bbox_), kps);
        if(kps.size() > MAX_TRACK_POINTS_NUM_)
        {
            cv::KeyPointsFilter::retainBest(kps, MAX_TRACK_POINTS_NUM_);
        }
        for(auto kp:kps)
        {
            kp.pt.x += bbox_.x;
            kp.pt.y += bbox_.y;
            track_points_.push_back(kp.pt);
        }
    }
    
    // stop tracking if too few tracking points
    if(track_points_.size() < 2)
    {
        status_ = false;
    }
    
    std::cout << "updating lktracker done" <<std::endl;
}

cv::Ptr<cv::FastFeatureDetector> LkTracker::detector_ = cv::FastFeatureDetector::create();

TrackerManager::TrackerManager(cv::Mat _frame, std::vector<cv::Rect> _rois):
    ids_(0),
    COST_THRESHOLD_(100),
    USE_KF_(true)
{
    for(auto roi:_rois)
    {
        auto tracker_ptr = new LkTracker(_frame, roi, ids_, USE_KF_);
        tracker_ptrs_.push_back(tracker_ptr);
        ids_ = (ids_+1)%100000;
    }
    _frame.copyTo(last_frame_);
        
}

std::vector<cv::Rect> TrackerManager::getAllBox()
{
    std::vector<cv::Rect> recs;
    for(auto tracker_ptr : tracker_ptrs_)
    {
        if(tracker_ptr->accepted_)
        {
            recs.push_back(tracker_ptr->getBbox());
        }
    }
    return recs;
}

std::vector<cv::Scalar> TrackerManager::getAllColor()
{
    std::vector<cv::Scalar> colors;
    for(auto tracker_ptr : tracker_ptrs_)
    {
        if(tracker_ptr->accepted_)
        {
            colors.push_back(tracker_ptr->getColor());
        }
    }
    return colors;
}


std::vector<cv::Point2f> TrackerManager::getAllPoints()
{
    return all_new_points_;
}

void TrackerManager::updateTrackersWithNewFrame(const cv::Mat& _frame)
{
    // Update all tracking points
    //int tracker_num = tracker_ptrs_.size();

    std::vector<cv::Point2f> all_old_points;
    //std::vector<cv::Point2f> all_new_points;
    all_new_points_.clear();

    for(auto tracker_ptr:tracker_ptrs_)
    {
        for(auto iter=tracker_ptr->track_points_.begin(); iter != tracker_ptr->track_points_.end(); iter ++)
        {
            all_old_points.push_back(*iter);
        }
    }

    std::vector<unsigned char> status;
    std::vector<float> error;
    std::cout << "old points num: " << all_old_points.size() << std::endl;
    if(all_old_points.size() >= 2)
    {
        cv::calcOpticalFlowPyrLK(last_frame_, _frame, all_old_points, all_new_points_, status, error);
        int index = 0;
        for(auto tracker_ptr:tracker_ptrs_)
        {
            tracker_ptr->old_track_points_.clear();
            for(auto iter = tracker_ptr->track_points_.begin(); iter != tracker_ptr->track_points_.end(); index ++)
            {
                if(status[index] == 0)
                {
                    iter = tracker_ptr->track_points_.erase(iter);
                    continue;
                }
                tracker_ptr->old_track_points_.push_back(*iter);
                *iter = all_new_points_[index];
                iter ++;
            }
        }
    }
   
    // Update all tracking boxes 
    for(auto tracker_ptr:tracker_ptrs_) 
    {
        if(tracker_ptr->track_points_.size() < 2)
        {
            tracker_ptr->status_ = false;
        }
        if(tracker_ptr->getStatus())
        {
            try
            {
                tracker_ptr->updateLkTracker(_frame);
            }
            catch(std::invalid_argument& e)
            {
                std::cerr << e.what() << std::endl;
            }
        }
    }


    std::cout << "remove status false trackers" << std::endl;
    // remove rejected trackers
    auto old_tracker_ptrs = tracker_ptrs_;
    tracker_ptrs_.clear();
    for(auto tracker_ptr:old_tracker_ptrs)
    {
        if(!tracker_ptr->getStatus())
        {
            delete tracker_ptr;
        }
        else
        {
            tracker_ptrs_.push_back(tracker_ptr);
        }
    }

    // update the last_frame_
    _frame.copyTo(last_frame_);
}

int TrackerManager::getMatchingScore(const cv::Rect _rec1, const cv::Rect _rec2)
{
    // score = (1 - iou) * dx/width * dy/height * 100

    auto max = [](int a, int b){return a>b?a:b;};
    auto min = [](int a, int b){return a<b?a:b;};

    int xA = max(_rec1.x, _rec2.x);
    int yA = max(_rec1.y, _rec2.y);
    int xB = min(_rec1.x+_rec1.width, _rec2.x+_rec2.width);
    int yB = max(_rec1.y+_rec1.height, _rec2.y+_rec2.height);
    int interArea = 0;
    if(xB <= xA || yB <= yA)
    {
        interArea = 0;
    }
    else
    {
        interArea = (xB - xA +1)*(yB - yA +1);
    }
    int boxAArea = (_rec1.width+1)*(_rec1.height+1);
    int boxBArea = (_rec2.width+1)*(_rec2.height+1);
    float iou = float(interArea) / float(boxAArea + boxBArea - interArea);

    int x1 = _rec1.x + _rec1.width/2;
    int x2 = _rec2.x + _rec2.width/2;
    int y1 = _rec1.y + _rec1.height/2;
    int y2 = _rec2.y + _rec2.height/2;
    auto abs = [](float x){return x>0?x:-x;};
    float dx = abs(static_cast<float>(x2-x1));
    float dy = abs(static_cast<float>(y2-y1));


    int score = static_cast<int>((1.0-iou) * dx * dy * 100.0 / (_rec1.width * _rec1.height));

    //std::cout << "rec1" << std::endl << _rec1 << std::endl;
    //std::cout << "rec2" << std::endl << _rec2 << std::endl;
    //std::cout << "score:" << score << std::endl;
    
    if(score == 0)
    {
        score = 1;
    }
    return score;
}

bool TrackerManager::updateTrackersWithNewDetectionResults(const std::vector<cv::Rect>& _dets)
{
    // matching trackers with detection results
    int trackers_num = tracker_ptrs_.size();
    int dets_num = _dets.size();

    std::cout << "trackers: " << trackers_num << " dets: " << dets_num << std::endl;

    cv::Mat_<int> old_matrix;
    cv::Mat_<int> new_matrix;

    if(trackers_num && dets_num){

        std::cout << "initialize cost_matrix" << std::endl;
        cv::Mat_<int> cost_matrix(trackers_num, dets_num);
        int i = 0;
        for(auto tracker_ptr:tracker_ptrs_)
        {
            int j = 0;
            for(auto det:_dets)
            {
                std::cout <<"get matching score (" << i <<","<<j <<")=" ;
                cost_matrix(i, j) = getMatchingScore(tracker_ptr->getBbox(), det);
                std::cout << (int)cost_matrix(i, j) << std::endl;
                j ++;
            }
            i ++;
        }
    
        old_matrix = cost_matrix.clone();
        Munkres m;
        std::cout << "start hungarian " << trackers_num << " x " << dets_num << std::endl;
        std::cout << cost_matrix << std::endl;
        m.solve(cost_matrix);
        std::cout << "hungarian end" << std::endl;
        new_matrix = cost_matrix.clone();
    }


    std::vector<int> matched_dets(dets_num, 0);

    for(int i = 0; i < trackers_num; i ++)
    {
        bool matched = false;
        for(int j = 0; j < dets_num; j ++)
        {
            
            if(new_matrix(i, j) == 0)
            {
                if(old_matrix(i, j) < COST_THRESHOLD_) // even if it's matched, if the cost is too high, we don't use it
                {
                    matched_dets[j] = 1;
                    matched = true;
                    tracker_ptrs_[i] -> missing_frames_ = 0;
                    tracker_ptrs_[i] -> bbox_ = _dets[j];
                    if(USE_KF_)
                    {
                        tracker_ptrs_[i] -> KalmanUpdate(_dets[i]);
                    }
                    std::cout << "tracker " << i << " matched with detector " << j << std::endl;
                    if(!tracker_ptrs_[i]->accepted_)
                    {
                        tracker_ptrs_[i] -> getting_frames_ ++;
                        if(tracker_ptrs_[i] -> getting_frames_ > tracker_ptrs_[i]->MIN_ACCEPT_FRAMES_)
                        {
                            tracker_ptrs_[i]->accepted_ = true;
                        }
                    }
                }
            }
        }
        if(!matched)
        {
            tracker_ptrs_[i] -> missing_frames_ ++;
            if(tracker_ptrs_[i]->missing_frames_ > tracker_ptrs_[i]->MIN_REJECT_FRAMES_)
            {
                tracker_ptrs_[i]->rejected_ = true;
            }
        }
    }
  
    // add new trackers
    std::cout << "add new trackers" << std::endl;
    int det_ind = 0;
    for(auto det:_dets)
    {
        if(!matched_dets[det_ind])
        {
            //std::cout << "new LkTracker " << current_frame_.cols << " " << det.width << " " << det.height << std::endl;
            auto tracker_ptr = new LkTracker(last_frame_, det, ids_, USE_KF_);
            std::cout << "new LkTracker get" << std::endl;
            tracker_ptrs_.push_back(tracker_ptr);
            ids_ = (ids_+1)%100000;
        }
        det_ind ++;
    }

    std::cout << "remove rejected trackers" << std::endl;
    // remove rejected trackers
    auto old_tracker_ptrs = tracker_ptrs_;
    tracker_ptrs_.clear();
    for(auto tracker_ptr:old_tracker_ptrs)
    {
        if(tracker_ptr->rejected_)
        {
            delete tracker_ptr;
        }
        else
        {
            tracker_ptrs_.push_back(tracker_ptr);
        }
    }

    return true;
}
