#ifndef PERSON_FOLLOWER_NODE_POSE_HPP_
#define PERSON_FOLLOWER_NODE_POSE_HPP_

#include "types.hpp"
#include "image.hpp"

#include <dynamic_reconfigure/server.h>
#include <person_follower/PoseEstimateConfig.h>

class Poser {
    public:
        Poser() = default;

        /// @brief Initialize and load everything
        void init();

        /// @brief Runs the YOLO model, and returns detections
        /// @note If annot_pub evaluates as `true`, an annotated image will be published
        PoseArrayMsg detect(const ImagePtrStruct& data,
                            const Intrinsics& intrinsics,
                            ros::Publisher& annot_pub);
        
        /// @brief Visualization markers showing position of person
        MarkerArrayMsg pose_center_markers(const PoseArrayMsg& poses) const;

        /// @brief Visualization markers showing positions of each keypoint
        MarkerArrayMsg keypoint_markers(const ImagePtrStruct& data,
                                        const Intrinsics& intrinsics, 
                                        const PoseArrayMsg& poses) const;

        /// @brief Get the roi's bounding box for computing embeddings
        cv::Rect bounding_box(const PoseArrayMsg& poses,
                              const cv::Size& image_size,
                              uint32_t target,
                              std::string& error_message) const;

        /// @brief Get roi for a single person - width and height 0 if too few good points
        cv::Rect crop_tight(const PoseMsg& pose) const;

    private:
        /// @brief Dynamic reconfigure callback
        void reconfigure(person_follower::PoseEstimateConfig& config, uint32_t level);

    public:
        // YOLO & SORT
        SortTracker               tracker;
        std::unique_ptr<YoloPose> yolo;

        // YOLO settings
        bool  log_model_time{false};
        float keypoint_conf{0.0f};
        float detect_iou{0.0f};
        float detect_conf{0.0f};
        
        // SORT settings
        float track_iou{0.0f};
        int   track_min_hits{0};
        int   track_max_age{0};

    private:
        std::unique_ptr<dynamic_reconfigure::Server<person_follower::PoseEstimateConfig>> server;
        dynamic_reconfigure::Server<person_follower::PoseEstimateConfig>::CallbackType f;
};



///////////////////////
// Load resources & set up dynamic reconfigure
void Poser::init() {
    // All config options nested under person_follower/pose/...
    ros::NodeHandle nh("~pose");

    // 1: Load models - if using TensorRT, this may take up to 10 minutes
    //    the first time, but should be faster on subsequent iterations
    //    as cacheing is on - cache at '~/.ros/trt_cache'
    std::string model  = nh.param<std::string>("/yolo_model", "yolov8n-pose.onnx");
    std::string device = nh.param<std::string>("/device", "gpu");

    ROS_INFO("loading YOLO - this may be extra long the first time");
    yolo = std::make_unique<YoloPose>(person_follower_dir("models/" + model), device);
    ROS_INFO("YOLO done loading");

    server = std::make_unique<dynamic_reconfigure::Server<person_follower::PoseEstimateConfig>>(nh);
    f = boost::bind(&Poser::reconfigure, this, _1, _2);
    server->setCallback(f);
}

// Detect instances on camera information
PoseArrayMsg Poser::detect(const ImagePtrStruct& data, const Intrinsics& intrinsics, ros::Publisher& annot) {
    Clock::time_point start = Clock::now();

    auto detections = yolo->detect(
        data.rgb->image,
        detect_conf,
        detect_iou
    );

    if ( log_model_time ){
        long us = std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - start
        ).count();
        ROS_INFO("person_follower/pose: YOLO took %lu us for inference", us);
    }

    auto tracking_info = tracker.update(
        boxes(detections),
        track_iou,
        track_min_hits,
        track_max_age
    );

    PoseArrayMsg out;
    out.header = data.depth->header;
    out.format = "COCO-17";
    out.poses.reserve(detections.size());
    
    for ( size_t i = 0; i < detections.size(); i++ ) {
        PoseMsg pose;
        pose.keypoints.reserve(17);
        
        pose.id    = tracking_info.ids[i];
        pose.score = detections[i].score;

        static const std::vector<std::pair<size_t, size_t>> proposals = {
            {5, 12},  //< Left hip to right shoulder
            {11, 6},  //< Left shoulder to right hip
            {5, 6},   //< Left hip to right hip
            {11, 12}, //< Left shoulder to right shoulder
            {3, 4},   //< Left ear to right ear
        };

        for ( const auto& p: proposals ) {
            const auto& a = detections[i].keypoints[p.first];
            const auto& b = detections[i].keypoints[p.second];

            if ( a.conf > keypoint_conf && b.conf > keypoint_conf ) {
                const float x = (a.x + b.x) / 2.0f;
                const float y = (a.y + b.y) / 2.0f;

                const int u = static_cast<int>(x);
                const int v = static_cast<int>(y);
                const float z = data.z(u, v);

                const auto pt = intrinsics.world(u, v, z);
                pose.pose.position.x = pt.x;
                pose.pose.position.y = pt.y;
                pose.pose.position.z = pt.z;
            }
        }

        for ( const auto& kpt: detections[i].keypoints ) {
            KeypointMsg keypoint;
            keypoint.x    = kpt.x;
            keypoint.y    = kpt.y;
            keypoint.z    = kpt.z;
            keypoint.conf = kpt.conf;

            pose.keypoints.push_back(keypoint);
        }

        out.poses.push_back(pose);
    }

    // Draw
    if ( annot && annot.getNumSubscribers() ) {
        cv_bridge::CvImagePtr frame = boost::make_shared<cv_bridge::CvImage> (
            data.rgb->header,
            data.rgb->encoding, 
            data.rgb->image.clone()
        );

        // 1: Draw skeletons
        yolo->draw(frame->image, detections, false);

        // 2: Label ids
        for ( size_t i = 0; i < detections.size(); i++ ) {
            cv::Scalar boxcolor(50, 50, 50);

            if ( out.poses[i].id ) {
                boxcolor = cv::Scalar(50, 200, 50);
                cv::putText(
                    frame->image, 
                    cv::format("%d", out.poses[i].id),
                    detections[i].bbox.tl(), 
                    cv::FONT_HERSHEY_COMPLEX, 
                    1, 
                    cv::Scalar(200, 50, 50), 
                    2
                );    
            }

            // cv::rectangle(frame->image, cv::Rect(detections[i].bbox), boxcolor);
        }

        // 3: Draw embedding rois
        for ( const auto& p: out.poses ) {
            cv::Rect roi = crop_tight(p);
            if ( roi.width > 0 )
                cv::rectangle(frame->image, roi, cv::Scalar(50, 50, 50));
        }

        annot.publish(frame->toImageMsg());
    }

    return out;
}

// Dynamic reconfigure of settings
void Poser::reconfigure(person_follower::PoseEstimateConfig& config, uint32_t level) {
    log_model_time = config.log_model_time;

    if ( level & 1 ) {
        keypoint_conf = config.keypoint_conf;
        ROS_INFO("person_follower/yolo: keypoint  (conf - %.2f)", keypoint_conf);
    }

    if ( level & 2 ) {
        detect_iou = config.detect_iou;
        detect_conf = config.detect_conf;
        ROS_INFO(
            "person_follower/yolo: detection (iou - %.2f | conf - %.2f)",
            detect_iou,
            detect_conf
        );
    }

    if ( level & 4 ) {
        track_iou = config.track_iou;
        track_min_hits = config.track_min_hits;
        track_max_age = config.track_max_age;
        ROS_INFO(
            "person_follower/yolo: tracking  (iou - %.2f | min_hits - %d | max_age - %d)",
            track_iou,
            track_min_hits,
            track_max_age
        );
    }
}

// Mark reference point for position from camera
MarkerArrayMsg Poser::pose_center_markers(const PoseArrayMsg& poses) const {
    constexpr float DIAM = 0.05;
        visualization_msgs::MarkerArray markers;

        for ( size_t i = 0; i < poses.poses.size(); i++ ) {
            const auto& p = poses.poses[i];

            if ( p.pose.position.z <= 0.0f )
                continue;

            visualization_msgs::Marker marker;
            marker.header = poses.header;
            marker.lifetime = ros::Duration(0.1);

            marker.id = i * 2;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = p.pose.position.x;
            marker.pose.position.y = p.pose.position.y;
            marker.pose.position.z = p.pose.position.z;

            marker.pose.orientation.x = -0.70710678f;
            marker.pose.orientation.y = 0.0f;
            marker.pose.orientation.z = 0.0f;
            marker.pose.orientation.w = 0.70710678f;

            marker.scale.x = DIAM;
            marker.scale.y = DIAM;
            marker.scale.z = DIAM;

            marker.color.a = 1.0;
            marker.color.g = 0.8;

            markers.markers.push_back(marker);

            visualization_msgs::Marker text;
            text.header = poses.header;
            text.lifetime = ros::Duration(0.1);

            text.id = i * 2 + 1;
            text.text = std::to_string(poses.poses[i].id);
            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::Marker::ADD;

            text.pose.position.x = p.pose.position.x;
            text.pose.position.y = p.pose.position.y - 0.1;
            text.pose.position.z = p.pose.position.z - 0.1;

            text.pose.orientation.x = -0.70710678f;
            text.pose.orientation.y = 0.0f;
            text.pose.orientation.z = 0.0f;
            text.pose.orientation.w = 0.70710678f;

            text.scale.x = DIAM;
            text.scale.y = DIAM;
            text.scale.z = DIAM;

            text.color.a = 1.0;
            text.color.g = 0.6;
            text.color.r = 0.2;

            markers.markers.push_back(text);
        }

        return markers;
}

// Mark all joint locations, in 3D
MarkerArrayMsg Poser::keypoint_markers(const ImagePtrStruct& data,
                                       const Intrinsics& intrinsics, 
                                       const PoseArrayMsg& poses) const {
    constexpr float DIAM = 0.05;
    visualization_msgs::MarkerArray markers;

    for ( size_t i = 0; i < poses.poses.size(); i++ ) {
        for ( size_t k = 0; k < 17; k++ ) {
            const auto& kpt = poses.poses[i].keypoints[k];
            if ( kpt.conf < keypoint_conf )
                continue;

            visualization_msgs::Marker marker;
            marker.header   = data.depth->header;
            marker.lifetime = ros::Duration(0.1);

            marker.id = i * 17 + k;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            const int u = static_cast<int>(kpt.x);
            const int v = static_cast<int>(kpt.y);
            const float z = data.z(u, v);

            cv::Point3f pos = intrinsics.world(kpt.x, kpt.y, z);
            marker.pose.position.x = pos.x;
            marker.pose.position.y = pos.y;
            marker.pose.position.z = pos.z;

            marker.pose.orientation.x = -0.70710678f;
            marker.pose.orientation.y = 0.0f;
            marker.pose.orientation.z = 0.0f;
            marker.pose.orientation.w = 0.70710678f;

            marker.scale.x = DIAM;
            marker.scale.y = DIAM;
            marker.scale.z = DIAM;

            marker.color.a = 1.0;
            marker.color.b = 0.8;

            markers.markers.push_back(marker);
        }
    }

    return markers;
}

// Get bounding box of target if possible
cv::Rect Poser::bounding_box(const PoseArrayMsg& poses, 
                             const cv::Size& image_size,
                             uint32_t target, 
                             std::string& error_message) const {
    constexpr float aspect_ratio = 0.5f; // width / height

    for ( const auto& p: poses.poses ) {
        if ( p.id == target ) {
            cv::Rect tight = crop_tight(p);
            
            if ( tight.width == 0 ) {
                error_message = "not enough visible points";
                return tight;
            }

            cv::Point center = (tight.tl() + tight.br()) / 2.0;

            int width  = std::max(static_cast<float>(tight.width), tight.height * aspect_ratio);
            int height = width / aspect_ratio;
            
            if ( height > image_size.height ) {
                error_message = "target is too close";
                return cv::Rect(0, 0, 0, 0);
            }

            cv::Rect roi = cv::Rect(
                std::max(center.x - width / 2, 0),
                std::max(center.y - height / 2, 0),
                width,
                width / aspect_ratio
            );

            roi.x = std::min(roi.x, image_size.width - width / 2);
            roi.y = std::min(roi.y, image_size.height - height / 2);

            return roi;
        }
    }

    error_message = "target id not found";
    return cv::Rect(0, 0, 0, 0);
}

// Get tight (non aspect-ratio conforming) roi for target
// Strategy: Use all keypoints from torso, head, and knees
cv::Rect Poser::crop_tight(const PoseMsg& pose) const {
    constexpr float x_scale = 1.4;
    constexpr float y_shift = -0.15;
    constexpr float y_scale = 1.3;

    // 1: Ensure at least one from each pair, and one from each side is visible
    static const std::vector<std::pair<size_t, size_t>> pairs {
        {3, 4},   // Ears
        {5, 6},   // Shoulders
        {11, 12}, // Hips
        {13, 14}  // Knees
    };

    float x_min = std::numeric_limits<float>::max();
    float y_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_max = std::numeric_limits<float>::lowest();

    bool found_left  = false;
    bool found_right = false;

    for ( const auto& p: pairs ) {
        const auto& left = pose.keypoints[p.first];
        const auto& right = pose.keypoints[p.second];
        bool found_either = false;

        if ( left.conf > keypoint_conf ) {
            found_left = true;
            found_either = true;
            x_min = std::min(left.x, x_min);
            x_max = std::max(left.x, x_max);
            y_min = std::min(left.y, y_min);
            y_max = std::max(left.y, y_max);
        }

        if ( right.conf > keypoint_conf ) {
            found_right = true;
            found_either = true;
            x_min = std::min(right.x, x_min);
            x_max = std::max(right.x, x_max);
            y_min = std::min(right.y, y_min);
            y_max = std::max(right.y, y_max);
        }

        if ( !found_either )
            return cv::Rect(0, 0, 0, 0);
    }

    if ( !found_left || !found_right )
        return cv::Rect(0, 0, 0, 0);

    float width = (x_max - x_min) * x_scale;
    float height = (y_max - y_min) * y_scale;

    float cx = (x_max + x_min) / 2;
    float cy = (y_max + y_min) / 2 + height * y_shift;

    return cv::Rect(
        cx - width / 2,
        cy - height / 2, 
        width,
        height
    );
}   

#endif