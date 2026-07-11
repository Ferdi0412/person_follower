#ifndef PERSON_FOLLOWER_SIMPLE_HANDS_WRAPPER_HPP_
#define PERSON_FOLLOWER_SIMPLE_HANDS_WRAPPER_HPP_

#include "person_follower/mediapipe/mediapipe.h"
#include "person_follower/tracking/assignment.hpp"

#include <ros/ros.h>

#include <person_follower_msgs/PoseArray.h>
#include <person_follower_msgs/Pose.h>
#include <person_follower_msgs/Keypoint.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "PoseConfigServer.hpp"
#include "IntrinsicsSubscriber.hpp"

#include <chrono>

/// @brief Runs palm detection on every frame, and hand detection for first n instances
struct SimpleHandsWrapper {
    using Clock = std::chrono::high_resolution_clock;
    using PoseArrayMsg = person_follower_msgs::PoseArray;
    using PoseMsg      = person_follower_msgs::Pose;
    using KeypointMsg  = person_follower_msgs::Keypoint;
    
    /// Process image and depth and pose info
    void process(const cv_bridge::CvImageConstPtr& rgb,
                 const cv_bridge::CvImageConstPtr& depth,
                 const PoseArrayMsg& poses,
                 const IntrinsicsSubscriber& intrinsics,
                 PoseArrayMsg& out,
                 cv_bridge::CvImagePtr& annot) {
        const Clock::time_point start = Clock::now();

        auto palms = palm_detector->detect(
            rgb->image,
            mp_settings.detection_conf,
            mp_settings.detection_iou
        );

        std::vector<person_follower::mediapipe::MPHand> hands;
        for ( const auto d: palms ) {
            person_follower::mediapipe::MPHand hand = hand_detector->detect(rgb->image, d);
            if ( hand.score > mp_settings.conf )
                hands.push_back(hand);
        }

        if ( mp_settings.log_infer_time ) {
            long us = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start
            ).count();
            ROS_INFO("person_follower/pose: MediaPipe palm+hand took %lu us for inference", us);
        }

        associate(out, hands, poses, depth, intrinsics);

        if ( annot ) {
            palm_detector->draw(annot->image, palms);

            for ( size_t i = 0; i < hands.size(); i++ ) {
                cv::Scalar boxcolor(50, 50, 50);

                hand_detector->draw(annot->image, hands[i]);

                if ( out.poses[i].id ) {
                    boxcolor = cv::Scalar(50, 200, 50);
                    cv::putText(
                        annot->image, 
                        cv::format("%d", out.poses[i].id),
                        hands[i].landmarks[12], 
                        cv::FONT_HERSHEY_COMPLEX, 
                        1, 
                        cv::Scalar(50, 50, 50), 
                        2
                    );
                }
            }
        }

        if ( mp_settings.log_full_time ) {
            long us = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start
            ).count();
            ROS_INFO("person_follower/pose:      took %lu us for callback", us);
        }
    }

    /// Use shortest distance assignment to match hand to wrist
    void associate(PoseArrayMsg& out,
                   const std::vector<person_follower::mediapipe::MPHand>& hands,
                   const PoseArrayMsg& poses,
                   const cv_bridge::CvImageConstPtr& depth,
                   const IntrinsicsSubscriber& intrinsics) {
        // 1: Assign body/pose ids to each hand
        std::vector<uint32_t> ids;
        std::vector<cv::Point2f> pose_wrists;
        
        for ( const auto& p: poses.poses ) {
            const auto& l_wrist = p.keypoints[9];
            const auto& r_wrist = p.keypoints[10];
            if ( l_wrist.conf > mp_settings.conf ) {
                pose_wrists.push_back(cv::Point2f(l_wrist.x, l_wrist.y));
                ids.push_back(p.id);
            }

            if ( r_wrist.conf > mp_settings.conf ) {
                pose_wrists.push_back(cv::Point2f(r_wrist.x, r_wrist.y));
                ids.push_back(p.id);
            }
        }

        std::vector<cv::Point2f> hand_wrists;
        for ( const auto& h: hands )
            hand_wrists.push_back(h.landmarks[0]);

        std::vector<size_t> matches = person_follower::tracking::hungarian(
            hand_wrists,
            pose_wrists,
            50.0f 
            /// @todo - tune 50 pixel limit...
        );

        // 2: Estimate position for any hands facing camera 
        out.header = depth->header;
        out.format = "MP-Hand";
        out.poses.reserve(hands.size());

        for ( size_t i = 0; i < hands.size(); i++ ) {
            const auto& hand = hands[i];

            PoseMsg handpose;
            handpose.keypoints.reserve(17);

            handpose.id = (matches[i] > ids.size()) ? 0 : ids[matches[i]];
            
            handpose.score = hand.right_handed;

            // Check if palm is aligned with camera's central axis
            cv::Point3f v1 = hand.world_landmarks[5] - hand.world_landmarks[0];
            cv::Point3f v2 = hand.world_landmarks[13] - hand.world_landmarks[0];

            cv::Point3f normal = v1.cross(v2);

            float norm = cv::norm(normal);
            if ( norm > 1e-8f )
                normal /= norm;

            // Check if within 60 degrees of camera axis
            if ( std::abs(normal.dot(cv::Point3f{0, 0, 1})) > 0.5 ) {
                const cv::Point2f& middle_mcp = hand.landmarks[9];
                
                const cv::Point palm_center = (middle_mcp + hand.landmarks[0]) / 2;
                const float z = depth->image.at<uint16_t>(palm_center.y, palm_center.x) * 0.001;

                auto pos = intrinsics.world(middle_mcp.x, middle_mcp.y, z);

                handpose.pose.position.x = pos.x;
                handpose.pose.position.y = pos.y;
                handpose.pose.position.z = pos.z;
            }

            for ( const cv::Point3f& kpt: hand.world_landmarks ) {
                KeypointMsg keypoint;
                keypoint.x = kpt.x;
                keypoint.y = kpt.y;
                keypoint.z = kpt.z;

                handpose.keypoints.push_back(keypoint);
            }

            out.poses.push_back(handpose);
        }
    }



    /// Display each landmark
    visualization_msgs::MarkerArray pose_visualization(const PoseArrayMsg& poses) const {
        constexpr float DIAM = 0.02;
        visualization_msgs::MarkerArray markers;

        for ( size_t i = 0; i < poses.poses.size(); i++ ) {
            const auto& p = poses.poses[i];

            if ( p.pose.position.z <= 0.0f )
                continue;

            for ( size_t j = 0; j < 21; j++ ) {
                const auto& k = p.keypoints[j];
                visualization_msgs::Marker marker;
                marker.header = poses.header;
                marker.lifetime = ros::Duration(0.1);

                marker.id = i * 21 + j;
                marker.type = visualization_msgs::Marker::SPHERE;
                marker.action = visualization_msgs::Marker::ADD;

                marker.pose.position.x = k.x + p.pose.position.x;
                marker.pose.position.y = k.y + p.pose.position.y;
                marker.pose.position.z = k.z + p.pose.position.z;

                marker.scale.x = DIAM;
                marker.scale.y = DIAM;
                marker.scale.z = DIAM;

                marker.color.a = 1.0;
                if ( poses.poses[i].score > 0.5 ) { // Right
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



    /// Setup everything
    void init(const std::string& device="gpu") {
        ros::NodeHandle args("~");

        mp_settings.init(ros::NodeHandle("~mediapipe"));

        std::string hand_model = args.param<std::string>(
            "/pose/hand_model", 
            person_follower::utils::person_follower_dir("models/mediapipe_hand.onnx")
        );

        std::string palm_model = args.param<std::string>(
            "/pose/palm_model",
            person_follower::utils::person_follower_dir("models/mediapipe_palm.onnx")
        );

        ROS_INFO("person_follower/Pose: setting up MediaPipe models - this might take some time");
        hand_detector = std::make_unique<person_follower::mediapipe::MPHandModel>(hand_model, device);
        palm_detector = std::make_unique<person_follower::mediapipe::MPPalmDetector>(palm_model, device);

        ROS_INFO("person_follower/pose: MediaPipe models loaded - first few detections might still be a little slow");
    }

    /// Dynamic reconfig class settings
    SimpleHandsWrapper()
     : mp_settings("person_follower (mediapipe)", 0.36f, 0.3f, 0.65f) {}

    // Dynamic reconfig settings
    PoseConfigServer mp_settings;

    // Resources used in main loop
    std::unique_ptr<person_follower::mediapipe::MPHandModel> hand_detector;
    std::unique_ptr<person_follower::mediapipe::MPPalmDetector> palm_detector;
};

#endif