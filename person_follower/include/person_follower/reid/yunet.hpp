#ifndef FOLLOWER_REID_YUNET_HPP_
#define FOLLOWER_REID_YUNET_HPP_

#include "person_follower/core/core.h"
#include "person_follower/pose/base.hpp"

namespace person_follower {
    namespace reid {
        using FaceDetection = pose::PoseDetection;
        using Keypoint = pose::Keypoint;
        using pose::boxes;

        /**
         * @brief Detect faces using the YuNet model
         */
        class YuNet: public core::ModelBase {
            public:
                /// @brief Load model and assert the appropriate shapes
                YuNet(const std::string& model, 
                      const std::string& device = "gpu",  
                      int threads = 0,
                      const std::string& cache_dir = "")
                 : core::ModelBase(model, device, threads, cache_dir) {
                    session_assert_inputs("YuNet", model, {{1, 3, 640, 640}});
                    session_assert_outputs("YuNet", model, {
                        {1, 6400, 1},
                        {1, 1600, 1},
                        {1, 400,  1},
                        {1, 6400, 1},
                        {1, 1600, 1},
                        {1, 400,  1},
                        {1, 6400, 4},
                        {1, 1600, 4},
                        {1, 400,  4},
                        {1, 6400, 10},
                        {1, 1600, 10},
                        {1, 400,  10}
                    });
                }

                /// @brief Process image and output detected faces
                std::vector<FaceDetection> detect(const cv::Mat& image,
                                                  float score = 0.9f,
                                                  float nms = 0.3f) {
                    const std::vector<int64_t> shape = {1, 3, 640, 640};

                    prepare(image);
                    preprocess(image);
                    Ort::Value tensor = session_tensor(chw.data(), shape);
                    std::vector<Ort::Value> result = session_run(tensor);
                    return postprocess(result, score, nms);
                }

                /// @brief Extract detection from image
                static cv::Mat extract(const cv::Mat& image, const FaceDetection& detection, const cv::Size out_size) {
                    cv::Mat aligned;
                    align(image, detection, aligned, out_size);
                    return aligned;
                }

                /// @brief Draw results
                static void draw(cv::Mat& image, const FaceDetection& detection) {
                    static const cv::Scalar boxcolor(100, 50, 50);
                    static const cv::Scalar pointcolor(100, 50, 0);
                    
                    cv::rectangle(image, cv::Rect(detection.bbox), boxcolor, 2);

                    for ( const auto& k: detection.keypoints )
                        cv::circle(image, cv::Point(k.x, k.y), 2, pointcolor, 2);
                }

                /// @brief Draw results
                static void draw(cv::Mat& image, const std::vector<FaceDetection>& detections) {
                    for ( auto f: detections ) {
                        draw(image, f);
                    }
                }

            private:
                cv::Size image_size{0, 0};
                float ratio = 0.0f;
                cv::Size roi{0, 0};
                cv::Mat blob;
                std::vector<float> chw;

                /// @brief Landmark template to compute affine transform
                static cv::Mat landmark_template(const cv::Size& out_size) {
                    std::vector<cv::Point2f> pts = {
                        {38.2946f / 112.0f * out_size.width, 51.6963f / 112.0f * out_size.height},   // right eye
                        {73.5318f / 112.0f * out_size.width, 51.5014f / 112.0f * out_size.height},   // left eye
                        {56.0252f / 112.0f * out_size.width, 71.7366f / 112.0f * out_size.height},   // nose
                        {41.5493f / 112.0f * out_size.width, 92.3655f / 112.0f * out_size.height},   // right mouth corner
                        {70.7299f / 112.0f * out_size.width, 92.2041f / 112.0f * out_size.height}    // left mouth corner
                    };
                    return cv::Mat(pts, true).reshape(1, 5);
                }

                /// @brief Resize and align around head
                static void align(const cv::Mat& image, const FaceDetection& detection, cv::Mat& aligned, cv::Size aligned_size) {                    
                    if ( detection.keypoints.size() != 5 )
                        throw std::invalid_argument("YuNet: invalid face - YuNet detections have 5 keypoints");
                    
                    cv::Mat dst(5, 2, CV_32F);

                    for ( int i = 0; i < 5; i++ ) {
                        dst.at<float>(i, 0) = detection.keypoints[i].x;
                        dst.at<float>(i, 1) = detection.keypoints[i].y;
                    }

                    const auto reference = landmark_template(aligned_size);

                    // Strictest alignment
                    cv::Mat m = cv::estimateAffinePartial2D(dst, reference);
                    
                    // Slightly relaxed, still landmark based
                    if ( m.empty() )
                        m = cv::estimateAffine2D(dst, reference);

                    // Simply use bounding box
                    if ( m.empty() ) {
                        const float w = static_cast<float>(aligned_size.width);
                        const float h = static_cast<float>(aligned_size.height);

                        const cv::Point2f target_edges[3] = {
                            {0, h},
                            {0, 0},
                            {w, 0}
                        };

                        const float x = detection.bbox.x;
                        const float y = detection.bbox.y;
                        const float bw = detection.bbox.width;
                        const float bh = detection.bbox.height;

                        const cv::Point2f detection_bbox[3] = {
                            {x,      y + bh},
                            {x,      y},
                            {x + bw, y}
                        };

                        m = cv::getAffineTransform(detection_bbox, target_edges);
                    }

                    cv::warpAffine(image, aligned, m, aligned_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                }

                /// @brief Compute dimensions needed
                inline void prepare(const cv::Mat& image) {
                    if ( image.cols == image_size.width && image.rows == image_size.height )
                        return;

                    image_size = cv::Size(image.cols, image.rows);
                    ratio = std::min(640 / static_cast<float>(image.cols), 640 / static_cast<float>(image.rows));
                    roi = cv::Size(
                        std::round(image.cols * ratio),
                        std::round(image.rows * ratio)
                    );

                    blob = cv::Mat(640, 640, CV_8UC3);
                    blob.setTo(cv::Scalar(0, 0, 0));
                    chw = std::vector<float>(640 * 640 * 3);
                }

                /// @brief Preprocessing of image
                inline void preprocess(const cv::Mat& image) {
                    cv::resize(
                        image,
                        blob(cv::Rect(0, 0, roi.width, roi.height)),
                        roi,
                        0, 0, cv::INTER_LINEAR
                    );

                    const uint8_t* src = blob.ptr<uint8_t>();

                    float* b = chw.data() + 0 * 640 * 640;
                    float* g = chw.data() + 1 * 640 * 640;
                    float* r = chw.data() + 2 * 640 * 640;

                    for ( size_t i = 0; i < 640 * 640; i++ ) {
                        b[i] = src[3*i + 0]; // / 255.0f;
                        g[i] = src[3*i + 1]; // / 255.0f;
                        r[i] = src[3*i + 2]; // / 255.0f;
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

                /// @brief Decode one stride's set of tensors
                /// Don't worry about it - it works
                void decode_stride(
                    const float* cls,
                    const float* obj,
                    const float* bbox,
                    const float* kpts,
                    int stride,
                    const cv::Size& grid,
                    float scale,
                    std::vector<float>& score_vec,
                    std::vector<cv::Rect2f>& box_vec,
                    std::vector<std::vector<Keypoint>>& landmark_vec,
                    float score_threshold
                ) const {
                    // bool apply_logits = stride == 8;

                    for ( int row = 0; row < grid.height; row++ ) {
                        for ( int col = 0; col < grid.width; col++ ) {
                            int idx = row * grid.width + col;

                            // Sometimes it's needed, usually it's not - I have yet to find any
                            // documentation or others with this issue... just gotta pray
                            // float c = apply_logits ? sigmoid(cls[idx]) : cls[idx];
                            // float o = apply_logits ? sigmoid(obj[idx]) : obj[idx];
                            float c = cls[idx];
                            float o = obj[idx];
                            float score = std::sqrt(c * o);

                            if ( score < score_threshold )
                                continue;

                            float cx = (col + bbox[idx * 4]) * stride;
                            float cy = (row + bbox[idx * 4 + 1]) * stride;
                            float bw = std::exp(bbox[idx * 4 + 2]) * stride;
                            float bh = std::exp(bbox[idx * 4 + 3]) * stride;

                            float x = (cx - bw * 0.5f) / scale;
                            float y = (cy - bh * 0.5f) / scale;
                            float w = bw / scale;
                            float h = bh / scale;

                            score_vec.push_back(score);
                            box_vec.push_back(cv::Rect2f(x, y, w, h));
                            
                            std::vector<Keypoint> landmarks(5);
                            for ( int k = 0; k < 5; k++ ) {
                                float lx = (col + kpts[idx * 10 + k * 2]) * stride / scale;
                                float ly = (row + kpts[idx * 10 + k * 2 + 1]) * stride / scale;

                                landmarks[k] = Keypoint(lx, ly);
                            }
                            landmark_vec.push_back(landmarks);
                        }
                    }
                }

                /// @brief Apply postprocessing
                std::vector<FaceDetection> postprocess(const std::vector<Ort::Value>& tensors, 
                                                       float score_threshold, 
                                                       float nms_threshold) {
                    const int strides[3] = {8, 16, 32};

                    std::vector<FaceDetection> candidates;

                    std::vector<float> score_vec;
                    std::vector<cv::Rect2f> box_vec;
                    std::vector<std::vector<Keypoint>> landmark_vec;

                    for ( int s = 0; s < 3; s++ ) {
                        int stride = strides[s];
                        cv::Size grid(640 / stride, 640 / stride);

                        const float* cls  = tensors[0 + s].GetTensorData<float>();
                        const float* obj  = tensors[3 + s].GetTensorData<float>();
                        const float* bbox = tensors[6 + s].GetTensorData<float>();
                        const float* kpts = tensors[9 + s].GetTensorData<float>();

                        decode_stride(
                            cls, 
                            obj,
                            bbox, 
                            kpts, 
                            stride, 
                            grid, 
                            ratio, 
                            score_vec, 
                            box_vec, 
                            landmark_vec, 
                            score_threshold
                        );
                    }

                    std::vector<int> indices;
                    std::vector<FaceDetection> results;
                    core::nms_boxes(box_vec, score_vec, score_threshold, nms_threshold, indices);

                    for ( int i: indices )
                        results.push_back(FaceDetection(score_vec[i], box_vec[i], landmark_vec[i]));
                        
                    return results;
                }


        };
    }
}

#endif // FOLLOWER_REID_YUNET_HPP_