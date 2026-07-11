#ifndef PERSON_FOLLOWER_YOLO_WRAPPER_HPP_
#define PERSON_FOLLOWER_YOLO_WRAPPER_HPP_

#include "person_follower/pose/yolo.hpp"
#include "person_follower/tracking/sort.hpp"

#include <ros/ros.h>

#include <person_follower_msgs/PoseArray.h>
#include <person_follower_msgs/Pose.h>
#include <person_follower_msgs/Keypoint.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include "PoseConfigServer.hpp"
#include "TrackingConfigServer.hpp"
#include "IntrinsicsSubscriber.hpp"

#include <chrono>

/// @brief Encompases all functions for YOLO
/// 1: init()
/// 2: process(rgb, depth, poses_out, [annot]) - assumes poses_out.poses, and annot is not
struct YoloWrapper {
    using Clock = std::chrono::high_resolution_clock;
    using PoseArrayMsg = person_follower_msgs::PoseArray;
    using PoseMsg      = person_follower_msgs::Pose;
    using KeypointMsg  = person_follower_msgs::Keypoint;

    /// Process image and depth info
    void process(const cv_bridge::CvImageConstPtr& rgb,
                 const cv_bridge::CvImageConstPtr& depth,
                 const IntrinsicsSubscriber& intrinsics,
                 PoseArrayMsg& out,
                 cv_bridge::CvImagePtr& annot) {
        const Clock::time_point start = Clock::now();

        auto detections = yolo->detect(
            rgb->image, 
            yolo_settings.detection_conf, 
            yolo_settings.detection_iou
        );

        if ( yolo_settings.log_infer_time ) {
            long us = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start
            ).count();
            ROS_INFO("person_follower/pose: YOLO took %lu us for inference", us);
        }

        auto tracking_info = tracker.update(
            boxes(detections),
            tracker_settings.iou,
            tracker_settings.min_hits,
            tracker_settings.max_age
        );

        out.header = depth->header;
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

                if ( a.conf > yolo_settings.conf && b.conf > yolo_settings.conf ) {
                    const float x = (a.x + b.x) / 2.0f;
                    const float y = (a.y + b.y) / 2.0f;

                    const int u = static_cast<int>(x);
                    const int v = static_cast<int>(y);
                    const float z = depth->image.at<uint16_t>(v, u) * 0.001;

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

        if ( annot ) {
            yolo->draw(annot->image, detections);

            for ( size_t i = 0; i < detections.size(); i++ ) {
                cv::Scalar boxcolor(50, 50, 50);

                if ( out.poses[i].id ) {
                    boxcolor = cv::Scalar(50, 200, 50);
                    cv::putText(
                        annot->image, 
                        cv::format("%d", out.poses[i].id),
                        detections[i].bbox.tl(), 
                        cv::FONT_HERSHEY_COMPLEX, 
                        1, 
                        cv::Scalar(200, 50, 50), 
                        2
                    );    
                }

                cv::rectangle(annot->image, cv::Rect(detections[i].bbox), boxcolor);
            }
        }

        if ( yolo_settings.log_full_time ) {
            long us = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start
            ).count();
            ROS_INFO("person_follower/pose:      took %lu us for callback", us);
        }
    }



    /// Display poses with id
    visualization_msgs::MarkerArray pose_visualization(const PoseArrayMsg& poses) const {
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



    visualization_msgs::MarkerArray joint_visualization(const PoseArrayMsg& poses, 
                                                        const cv_bridge::CvImageConstPtr& depth,
                                                        const IntrinsicsSubscriber& intrinsics) const {
        constexpr float DIAM = 0.05;
        visualization_msgs::MarkerArray markers;

        for ( size_t i = 0; i < poses.poses.size(); i++ ) {
            for ( size_t k = 0; k < 17; k++ ) {
                const auto& kpt = poses.poses[i].keypoints[k];
                if ( kpt.conf < yolo_settings.conf )
                    continue;

                visualization_msgs::Marker marker;
                marker.header   = depth->header;
                marker.lifetime = ros::Duration(0.1);

                marker.id = i * 17 + k;
                marker.type = visualization_msgs::Marker::SPHERE;
                marker.action = visualization_msgs::Marker::ADD;

                const int u = static_cast<int>(kpt.x);
                const int v = static_cast<int>(kpt.y);
                const float z = depth->image.at<uint16_t>(v, u) * 0.001;

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



    /// Setup everything
    void init(const std::string& device="gpu") {
        ros::NodeHandle args("~");

        yolo_settings.init(ros::NodeHandle("~yolo"));
        tracker_settings.init(ros::NodeHandle("~sort"));

        std::string model = args.param<std::string>(
            "/pose/model", 
            person_follower::utils::person_follower_dir("models/yolov8n-pose.onnx")
        );

        ROS_INFO("person_follower/Pose: setting up YOLO model - this might take some time");
        yolo = std::make_unique<person_follower::pose::YoloPose>(model, device);
        ROS_INFO("person_follower/pose: YOLO model loaded - first few detections might still be a little slow");
    }


    /// Dynamic reconfig class settings
    YoloWrapper()
     : yolo_settings("person_follower (yolo)", 0.4f, 0.5f, 0.65f),
       tracker_settings("person_follower (yolo-sort)") {}



    // Dynamic reconfig settings
    PoseConfigServer     yolo_settings;
    TrackingConfigServer tracker_settings;

    // Resources used in main loop
    std::unique_ptr<person_follower::pose::YoloPose> yolo;
    person_follower::tracking::SortTracker           tracker;
};

#endif 