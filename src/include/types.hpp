#ifndef PERSON_FOLLOWER_NODE_TYPES_HPP_
#define PERSON_FOLLOWER_NODE_TYPES_HPP_

#include "person_follower/person_follower.h"

using person_follower::utils::person_follower_dir;
using person_follower::pose::YoloPose;
using person_follower::tracking::SortTracker;

using person_follower::mediapipe::MPPalm;
using person_follower::mediapipe::MPHand;
using person_follower::mediapipe::MPHandModel;
using person_follower::mediapipe::MPPalmDetector;

#include <person_follower_msgs/PoseArray.h>
#include <person_follower_msgs/Pose.h>
#include <person_follower_msgs/Keypoint.h>

using PoseArrayMsg = person_follower_msgs::PoseArray;
using PoseMsg      = person_follower_msgs::Pose;
using KeypointMsg  = person_follower_msgs::Keypoint;

#include <chrono>
using Clock = std::chrono::high_resolution_clock;

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

using MarkerArrayMsg = visualization_msgs::MarkerArray;
using MarkerMsg      = visualization_msgs::Marker;

#endif // PERSON_FOLLOWER_NODE_TYPES_HPP_