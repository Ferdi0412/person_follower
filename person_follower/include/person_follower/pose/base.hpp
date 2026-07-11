#ifndef PERSON_FOLLOWER_POSE_BASE_HPP_
#define PERSON_FOLLOWER_POSE_BASE_HPP_

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace person_follower {
    namespace pose {
        /**
         * @brief A single point in the output from the model
         */
        struct Keypoint {
            float x{0.0f};
            float y{0.0f};
            float z{0.0f};
            float conf{0.0f};

            Keypoint() = default;
            Keypoint(float x_, float y_, float z_ = 0.0f, float c = 1.0f)
             : x(x_), y(y_), z(z_), conf(c) {}
        };

        /**
         * @brief A single instance output by a pose model
         */
        struct PoseDetection {
            float score{0.0f};
            cv::Rect2f bbox;
            std::vector<Keypoint> keypoints;

            PoseDetection() = default;
            PoseDetection(float s, const cv::Rect2f& b, const std::vector<Keypoint>& k)
             : score(s), bbox(b), keypoints(k) {}
        };

        /// @brief Helper function to extract boxes from output
        inline std::vector<cv::Rect2f> boxes(const std::vector<PoseDetection>& detections) {
            std::vector<cv::Rect2f> b(detections.size());
            for ( size_t i = 0; i < detections.size(); i++ )
                b[i] = detections[i].bbox;
            return b;
        }

        /**
         * @brief Abstract base class for human pose estimation model
         */
        class PoseModel {
            protected: 
                PoseModel() = default;

            public:
                /// @brief Detect instances
                virtual std::vector<PoseDetection> detect(const cv::Mat& image, 
                                                          float score = 0.5f, 
                                                          float nms = 0.3f) = 0;

                /// @brief Draw a single instance
                virtual void draw(cv::Mat& image, const PoseDetection& pose, bool w_rect = false) const = 0;

                /// @brief Draw several instances
                virtual void draw(cv::Mat& image, const std::vector<PoseDetection>& poses, bool w_rect = false) const = 0;
        };
    }
}

#endif // FOLLOWER_POSE_BASE_HPP_