#ifndef PERSON_FOLLOWER_POSE_YOLO_HPP_
#define PERSON_FOLLOWER_POSE_YOLO_HPP_

#include "person_follower/core/core.h"
#include "person_follower/pose/base.hpp"

namespace person_follower {
    namespace pose {
        /**
         * @brief YOLO Pose models with NMS postprocessing (up to, not including 26)
         */
        class YoloPose: public PoseModel, public core::ModelBase {
            public:
                /// @brief Load model and assert the appropriate shapes
                YoloPose(const std::string& model, 
                         const std::string& device = "gpu", 
                         int threads = 0,
                        const std::string& cache_dir = "./trt_cache")
                 : core::ModelBase(model, device, threads, cache_dir) {
                    session_assert_inputs("YoloPose", model, {{1, 3, 640, 640}});
                    session_assert_outputs("YoloPose", model, {{1, 56, 8400}});
                }

                // Default destructor, no copying
                ~YoloPose() = default;
                YoloPose(const YoloPose&) = delete;
                YoloPose& operator=(const YoloPose&) = delete;

                /// @brief Run inference
                std::vector<PoseDetection> detect(const cv::Mat& image,
                                                  float score = 0.4f,
                                                  float nms = 0.5f) override {
                    static const std::vector<int64_t> shape = {1, 3, 640, 640};
                    prepare(image, cv::Scalar(114, 114, 114));
                    preprocess(image);
                    Ort::Value tensor = session_tensor(chw.data(), shape);
                    std::vector<Ort::Value> results = session_run(tensor);
                    return postprocess(results, score, nms);
                }

                /// @brief Draw a result instance
                void draw(cv::Mat& image, const PoseDetection& pose, bool w_rect = true) const override {
                    static const cv::Scalar boxcolor(100, 200, 50);
                    const auto& links = yolo_links();
                    const auto& colors = yolo_colors();

                    if ( w_rect )
                        cv::rectangle(image, cv::Rect(pose.bbox), boxcolor);

                    if ( pose.keypoints.size() != 17 )
                        return;

                    for ( size_t i = 0; i < links.size(); i++ ) {
                        const auto& a = pose.keypoints[links[i].first];
                        const auto& b = pose.keypoints[links[i].second];
                        cv::line(image, cv::Point(a.x, a.y), cv::Point(b.x, b.y), colors[i]);
                    }
                }

                /// @brief Draw all results
                void draw(cv::Mat& image, const std::vector<PoseDetection>& poses, bool w_rect = true) const {
                    for ( const auto& p: poses )
                        draw(image, p, w_rect);
                }

                /// @brief Links to draw 
                static const std::array<std::pair<size_t, size_t>, 18>& yolo_links() {
                    static const std::array<std::pair<size_t, size_t>, 18> links = {{
                        {0, 1},   {0, 2},   /*< nose-eye     */ {1, 3},   {2, 4},   //< eye-ear
                        {3, 5},   {4, 6},   /*< ear-shoulder */ {5, 7},   {6, 8},   //<shoulder-elbow
                        {7, 9},   {8, 10},  /*< elbow-wrist  */ {5, 11},  {6, 12},  //< shoulder-hip
                        {11, 13}, {12, 14}, /*< hip-knee     */ {13, 15}, {14, 16}, //< knee-ankle
                        {5, 6}, /*< shoulder-shoulder */ {11, 12} //<hip-hip
                    }};
                    return links;
                }

                /// @brief Colors per link 
                static const std::array<cv::Scalar, 18>& yolo_colors() {
                    static const cv::Scalar HEAD   = cv::Scalar(50, 200, 0);
                    static const cv::Scalar TORSO  = cv::Scalar(70, 100, 50);
                    static const cv::Scalar LOWER  = cv::Scalar(80, 70,  150); 
                    static const std::array<cv::Scalar, 18> colors = {{
                        HEAD,  HEAD,  HEAD,  HEAD,  HEAD,  HEAD,  //< nose-eye-ear-shoulder
                        TORSO, TORSO, TORSO, TORSO, TORSO, TORSO, //< shoulder-elbow-wrist-hip
                        LOWER, LOWER, LOWER, LOWER, TORSO, TORSO
                    }};
                    return colors;
                }

            private:
                cv::Size image_size{0, 0};
                float scale = 0.0f;
                cv::Size roi{0, 0};
                cv::Point2i bias{0, 0};
                cv::Mat blob;
                std::array<float, 640 * 640 * 3> chw;

                /// @brief Computation as needed
                inline void prepare(const cv::Mat& image, cv::Scalar color) {
                    if ( image.cols == image_size.width && image.rows == image_size.height )
                        return;

                    image_size = cv::Size(image.cols, image.rows);
                    scale = std::min(640 / static_cast<float>(image.cols), 640 / static_cast<float>(image.rows));
                    roi = cv::Size(image.cols * scale, image.rows * scale);
                    bias = cv::Point2i((640 - roi.width) / 2, (640 - roi.height) / 2);

                    // scale = std::min(640 / image.cols, 640 / image.rows);
                    // int w = std::lround(image.cols * scale);
                    // int h = std::lround(image.rows * scale);
                    // float dw = (640 - w) / 2.0f;
                    // float dh = (640 - h) / 2.0f;
                    // int x = std::lround(dw - 0.1f);
                    // int y = std::lround(dh - 0.1f);
                    // roi = cv::Size(w, h);
                    // bias = cv::Point2i(x, y);
                
                    blob = cv::Mat(640, 640, CV_8UC3);
                    blob.setTo(color);
                }

                /// @brief Preprocessing of image - resize, normalize, channel re-arrange
                inline void preprocess(const cv::Mat& image) {
                    cv::resize(
                        image,
                        blob(cv::Rect(bias.x, bias.y, roi.width, roi.height)),
                        roi,
                        0, 0, cv::INTER_LINEAR
                    );

                    const uint8_t* src = blob.ptr<uint8_t>();

                    float* r = chw.data();
                    float* g = chw.data() + 1 * 640 * 640;
                    float* b = chw.data() + 2 * 640 * 640;

                    for ( size_t i = 0; i < 640 * 640; i++ ) {
                        b[i] = src[3*i + 0] / 255.0f;
                        g[i] = src[3*i + 1] / 255.0f;
                        r[i] = src[3*i + 2] / 255.0f;
                    }
                }

                /// @brief Clamp value
                static inline float clamp(float min, float v, float max) {
                    return std::max(min, std::min(v, max));
                }

                /// @brief Apply sigmoid
                static inline float sigmoid(float v) {
                    return 1.0f / (1.0f + std::exp(-v));
                }

                /// @brief Apply postprocessing
                std::vector<PoseDetection> postprocess(const std::vector<Ort::Value>& tensors, float score_threshold, float nms_threshold) {
                    std::vector<PoseDetection> results;

                    std::vector<cv::Rect2f> boxes;
                    std::vector<float> scores;
                    std::vector<std::vector<Keypoint>> keypoints;
                    boxes.reserve(64);
                    scores.reserve(64);
                    keypoints.reserve(64);

                    const float* raw = tensors[0].GetTensorData<float>();
                    for ( size_t d = 0; d < 8400; d++ ) {
                        const float score = raw[4 * 8400 + d];
                        
                        if ( score < score_threshold )
                            continue;

                        const float cx = raw[0 * 8400 + d];
                        const float cy = raw[1 * 8400 + d];
                        const float w  = raw[2 * 8400 + d];
                        const float h  = raw[3 * 8400 + d];

                        const float x = clamp(0, (cx - w / 2.0f - bias.x) / scale, image_size.width - 1);
                        const float y = clamp(0, (cy - h / 2.0f - bias.y) / scale, image_size.height - 1);
                        cv::Rect2f box(
                            x,
                            y,
                            clamp(1, w / scale, image_size.width - x),
                            clamp(1, h / scale, image_size.height - y)
                        );

                        std::vector<Keypoint> kpts(17);
                        for ( int k = 0; k < 17; k++ ) {
                            const int offset = 5 + k * 3;
                            
                            const float x1 = ((raw[offset * 8400 + d] - bias.x) / scale);
                            const float y1 = ((raw[(offset + 1) * 8400 + d] - bias.y) / scale);
                            const float s1 = sigmoid(raw[(offset + 2) * 8400 + d]);
                            
                            kpts[k] = Keypoint(
                                clamp(0, x1, image_size.width - 1), 
                                clamp(0, y1, image_size.height - 1), 
                                0.0f,
                                s1
                            );
                        }

                        boxes.push_back(box);
                        scores.push_back(score);
                        keypoints.push_back(kpts);
                    }

                    if ( boxes.empty() )
                        return {};

                    std::vector<int> indices;
                    core::nms_boxes(boxes, scores, score_threshold, nms_threshold, indices);

                    results.reserve(indices.size());
                    for ( int idx: indices ) 
                        results.emplace_back(scores[idx], boxes[idx], keypoints[idx]);

                    return results;
                }
        };
    }
}

#endif // FOLLOWER_POSE_YOLO_HPP_