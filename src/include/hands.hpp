/// @note The hand-wrist allocation would be better with a distance-based hungarian for all
///       known wrists in the frame
#ifndef PERSON_FOLLOWER_NODE_HANDS_HPP_
#define PERSON_FOLLOWER_NODE_HANDS_HPP_

#include "types.hpp"
#include "image.hpp"

#include <dynamic_reconfigure/server.h>
#include <person_follower/HandEstimateConfig.h>

#include <std_msgs/Empty.h>

class Handser {
    public:
        Handser() = default;

        /// @brief Initialize and load everything
        void init();

        /// @brief Runs MediaPipe model, if target is non-zero
        /// @note If annot_pub evaluates as `true`, an annotated image will be published
        /// @param position is the last known position of the "center of mass"-ish of person
        PoseArrayMsg update(cv::Point3f& position,
                            const ImagePtrStruct& data, 
                            const Intrinsics& intrinsics,
                            const PoseArrayMsg& poses,
                            uint32_t target,
                            float keypoint_conf,
                            ros::Publisher& annot_pub);

        /// @brief Update both hands
        void update_both(const ImagePtrStruct& data, const PoseMsg& pose, float keypoint_conf);

        /// @brief Update a single hand
        void update_single(MPHand& hand, const ImagePtrStruct& data, const KeypointMsg& kpt, float z);

        /// @brief Update output message
        void append_hand(PoseArrayMsg& out, 
                         const MPHand& hand, 
                         const cv::Point3f& position,
                         bool is_right,
                         const ImagePtrStruct& data, 
                         const Intrinsics& intrinsics) const;

        /// @brief Visualization markers showing position of hand landmarks
        MarkerArrayMsg landmarker_markers(const PoseArrayMsg& hands, const cv::Point3f& position) const;

    private:
        /// @brief Dynamic reconfigure callback
        void reconfigure(person_follower::HandEstimateConfig& config, uint32_t level);

        // Communicate frequency of palm model
        ros::Publisher palm_ping;

        // MediaPipe
        std::unique_ptr<MPHandModel> hand_model;
        std::unique_ptr<MPPalmDetector> palm_model;

        // Hand self-tracking
        MPHand   left;
        MPHand   right;
        uint32_t current{0};

        // Model settings
        bool  log_model_time{false};
        float palm_iou{0.0f};
        float palm_conf{0.0f};
        float hand_conf{0.0f};
        float wrist_dist{0.0f};
        bool  project_points{false};

        std::unique_ptr<dynamic_reconfigure::Server<person_follower::HandEstimateConfig>> server;
        dynamic_reconfigure::Server<person_follower::HandEstimateConfig>::CallbackType f;
};



///////////////////////
// Load resources & set up dynamic reconfigure
void Handser::init() {
    // All config options nested under person_follower/hands/...
    ros::NodeHandle nh("~hands");

    // 1: Load models - if using TensorRT for the first time, this may
    //                  take up to 10 minutes
    std::string palm   = nh.param<std::string>("/palm_model", "mediapipe_palm.onnx");
    std::string hand   = nh.param<std::string>("/hand_model", "mediapipe_hand.onnx");
    std::string device = nh.param<std::string>("/device", "gpu");

    ROS_INFO("loading MediaPipe - this may be extra long the first time");
    palm_model = std::make_unique<MPPalmDetector>(person_follower_dir("models/" + palm), device);
    hand_model = std::make_unique<MPHandModel>(person_follower_dir("models/" + hand), device);
    ROS_INFO("MediaPipe done loading");

    server = std::make_unique<dynamic_reconfigure::Server<person_follower::HandEstimateConfig>>(nh);
    f = boost::bind(&Handser::reconfigure, this, _1, _2);
    server->setCallback(f);

    ros::NodeHandle gnh;
    palm_ping = gnh.advertise<std_msgs::Empty>("/person_follower/hands/palm_ping", 1);
}

// "Forward pass"
PoseArrayMsg Handser::update(cv::Point3f& position,
                             const ImagePtrStruct& data,
                             const Intrinsics& intrinsics,
                             const PoseArrayMsg& poses,
                             uint32_t target,
                             float keypoint_conf,
                             ros::Publisher& annot_pub) {
    if ( target != current ) {
        current = target;
        left.score  = 0;
        right.score = 0;
        position = cv::Point3f(0, 0, 0);
    }

    PoseArrayMsg out;
    out.header = data.depth->header;

    bool updated_left = false;
    bool updated_right = false;

    // Only need to update hands in the case that a target is specified
    if ( current ) {
        // Update last known position
        for ( const auto& p: poses.poses ) {
            if ( p.id == current ) {
                position.x = p.pose.position.x;
                position.y = p.pose.position.y;
                position.z = p.pose.position.z;
            }
        }

        // Update hands using previous position - quick path
        if ( left.score > hand_conf ) {
            left = hand_model->detect(data.rgb->image, left);
            updated_left = true;
        }

        if ( right.score > hand_conf ) {
            right = hand_model->detect(data.rgb->image, right);
            updated_right = true;
        }

        // Update hands using wrist location
        // Separate "update both" in case running palm x2 frequently is too slow
        if ( !updated_left && !updated_right ) {
            for ( const auto& p: poses.poses )
                if ( p.id == current )
                    update_both(data, p, keypoint_conf);
        }

        else if ( !updated_left || !updated_right ) {
            for ( const auto& p: poses.poses ) {
                if ( p.id != current )
                    continue;

                if ( !updated_left && p.keypoints[9].conf > keypoint_conf )
                    update_single(left, data, p.keypoints[9], p.pose.position.z);
                
                if ( !updated_right && p.keypoints[10].conf > keypoint_conf )
                    update_single(right, data, p.keypoints[10], p.pose.position.z);
            }
        }

        // Update pose array
        if ( left.score > hand_conf )
            append_hand(out, left, position, false, data, intrinsics);

        if ( right.score > hand_conf )
            append_hand(out, right, position, true, data, intrinsics);
    }

    // Draw
    if ( annot_pub && annot_pub.getNumSubscribers() ) {
        cv_bridge::CvImagePtr frame = boost::make_shared<cv_bridge::CvImage> (
            data.rgb->header,
            data.rgb->encoding, 
            data.rgb->image.clone()
        );

        if ( left.score > hand_conf )
            hand_model->draw(frame->image, left);

        if ( right.score > hand_conf )
            hand_model->draw(frame->image, right);

        cv::putText(
            frame->image,
            cv::format("Detecting hands for: %u", current),
            cv::Point(12, 30),
            cv::FONT_HERSHEY_COMPLEX,
            1,
            cv::Scalar(20, 20, 255),
            2
        );

        if ( current == 0 )
            cv::putText(
                frame->image,
                "To detect hands, set target id through '/person_follower/target'",
                cv::Point(0, frame->image.rows - 15),
                cv::FONT_HERSHEY_COMPLEX,
                1,
                cv::Scalar(20, 20, 255),
                2
            );

        for ( const auto& p: poses.poses ) {
            cv::putText(
                frame->image,
                cv::format("%u", p.id),
                cv::Point(p.keypoints[1].x, p.keypoints[1].y - 15),
                cv::FONT_HERSHEY_COMPLEX,
                1,
                cv::Scalar(20, 20, 255),
                2
            );
        }

        annot_pub.publish(frame->toImageMsg());
    }

    return out;
}

// Update both hands
void Handser::update_both(const ImagePtrStruct& data, const PoseMsg& pose, float keypoint_conf) {
    if ( pose.keypoints[9].conf > keypoint_conf )
        update_single(left, data, pose.keypoints[9], pose.pose.position.z);

    if ( pose.keypoints[10].conf > keypoint_conf )
        update_single(right, data, pose.keypoints[10], pose.pose.position.z);
}

// Update single hand
void Handser::update_single(MPHand& hand, const ImagePtrStruct& data, const KeypointMsg& kpt, float z) {
    cv::Rect roi(0, 0, data.rgb->image.cols, data.rgb->image.rows);
    cv::Mat img;

    // If beyond 1.5 meters, reduce the roi for detection for better quality
    if ( z > 1.5 ) {
        float width = z > 3.0 ? 192 : 382;

        roi = cv::Rect(
            kpt.x - width / 2,
            kpt.y - width / 2,
            width,
            width
        );

        roi.x = std::clamp(static_cast<float>(roi.x), 0.0f, data.rgb->image.cols - width);
        roi.y = std::clamp(static_cast<float>(roi.y), 0.0f, data.rgb->image.rows - width);
    }
    
    auto palms = palm_model->detect(data.rgb->image(roi), palm_conf, palm_iou);
    palm_ping.publish(std_msgs::Empty{});

    size_t matched = SIZE_MAX;
    float min_dist = wrist_dist;
    cv::Point2f target(kpt.x - roi.x, kpt.y - roi.y);

    for ( size_t i = 0; i < palms.size(); i++ ) {
        float dist = cv::norm(target - palms[i].landmarks[0]);

        if ( dist < min_dist ) {
            matched = i;
            min_dist = dist;
        }
    }

    if ( !(matched < palms.size()) )
        return;

    MPPalm& palm = palms[matched];

    // Important to update bounding box - otherwise no hit...
    palm.bbox.x += roi.x;
    palm.bbox.y += roi.y;

    for ( auto& kpt: palm.landmarks ) {
        kpt.x += roi.x;
        kpt.y += roi.y;
    }

    hand = hand_model->detect(data.rgb->image, palm);
}

// Dynamic reconfigure of settings
void Handser::reconfigure(person_follower::HandEstimateConfig& config, uint32_t level) {
    log_model_time = config.log_model_time;

    if ( level & 1 ) {
        palm_iou       = config.palm_iou;
        palm_conf      = config.palm_conf;
        ROS_INFO(
            "person_follower/hands: palm (iou - %.2f | conf - %.2f)",
            palm_iou,
            palm_conf
        );
    }

    if ( level & 2 ) {
        hand_conf      = config.hand_conf;
        ROS_INFO("person_follower/hands: hand (conf - %.2f)", hand_conf);
    }

    if ( level & 4 ) {
        wrist_dist = config.wrist_dist;
        ROS_INFO("person_follower/hands: detect (wrist_dist - %.2f)", wrist_dist);
    }

    if ( level & 8 ) {
        project_points = config.project_points;
        ROS_INFO(
            "person_follower/hads: project points - %s",
            project_points ? "True" : "False"
        );
    }
}

// Format PoseMsg for hand
void Handser::append_hand(PoseArrayMsg& out,
                          const MPHand& hand,
                          const cv::Point3f& position,
                          bool is_right, 
                          const ImagePtrStruct& data,
                          const Intrinsics& intrinsics) const {
    if ( hand.score < hand_conf )
        return;

    // First format all model outputs
    PoseMsg handpose;
    handpose.keypoints.reserve(21);
    handpose.id = current;
    handpose.score = is_right;

    for ( const cv::Point3f& kpt: hand.world_landmarks ) {
        KeypointMsg keypoint;
        keypoint.x = kpt.x;
        keypoint.y = kpt.y;
        keypoint.z = kpt.z;

        handpose.keypoints.push_back(keypoint);
    }
    
    // Then, check if hand aligned within 60 degree of camera axis
    // for good surface for depth estimation
    cv::Point3f v1 = hand.world_landmarks[5] - hand.world_landmarks[0];
    cv::Point3f v2 = hand.world_landmarks[13] - hand.world_landmarks[0];

    cv::Point3f normal = v1.cross(v2);

    float norm = cv::norm(normal);
    if ( norm > 1e-8f )
        normal /= norm;

    // constexpr ALIGN_THRESH = 0.5;
    constexpr float ALIGN_THRESH = 0.707107;

    if ( std::abs(normal.dot(cv::Point3f{0, 0, 1})) > ALIGN_THRESH ) {
        const cv::Point2f& middle_mcp = hand.landmarks[9];
                
        const cv::Point palm_center = (middle_mcp + hand.landmarks[0]) / 2;
        const float z = data.z(palm_center.x, palm_center.y);

        auto pos = intrinsics.world(middle_mcp.x, middle_mcp.y, z);

        // Keep position relative to person's "spine"-ish
        handpose.pose.position.x = pos.x - position.x;
        handpose.pose.position.y = pos.y;
        handpose.pose.position.z = pos.z - position.z;
    }

    out.poses.push_back(handpose);
}


MarkerArrayMsg Handser::landmarker_markers(const PoseArrayMsg& hands, const cv::Point3f& position) const {
    constexpr float DIAM = 0.02;
    visualization_msgs::MarkerArray markers;

    for ( size_t i = 0; i < hands.poses.size(); i++ ) {
        const auto& p = hands.poses[i];

        if ( p.pose.position.z <= 0.0f )
            continue;

        for ( size_t j = 0; j < 21; j++ ) {
            const auto& k = p.keypoints[j];
            visualization_msgs::Marker marker;
            marker.header = hands.header;
            marker.lifetime = ros::Duration(0.1);

            marker.id = i * 21 + j;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = k.x + p.pose.position.x + position.x;
            marker.pose.position.y = k.y + p.pose.position.y + position.y;
            marker.pose.position.z = k.z + p.pose.position.z + position.z;
            marker.pose.orientation.x = 0;
            marker.pose.orientation.y = 0;
            marker.pose.orientation.z = 0;
            marker.pose.orientation.w = 1;

            marker.scale.x = DIAM;
            marker.scale.y = DIAM;
            marker.scale.z = DIAM;

            marker.color.a = 1.0;
            if ( hands.poses[i].score > 0.5 ) { // Right
                marker.color.g = 0.6;
                marker.color.b = 0.4;
            }
            else {
                marker.color.b = 0.6;
                marker.color.g = 0.4;
            }
            
            markers.markers.push_back(marker);
        }
    }

        return markers;
}

#endif // PERSON_FOLLOWER_NODE_HANDS_HPP_