#ifndef PERSON_FOLLOWER_MEDIAPIPE_PALM_HPP_
#define PERSON_FOLLOWER_MEDIAPIPE_PALM_HPP_

#include "person_follower/core/core.h"
#include "person_follower/mediapipe/palm_anchors.hpp"

namespace person_follower {
    namespace mediapipe {
        /**
         * @brief Output of MediaPipe palm model
         * @note Landmarks are in order:
         *     0: Wrist, 1: IndexMCP, 2: MiddleMCP, 3: RingMCP
         *     4: PinkyMCP, 5: ThumbCMC, 6: ThumbMCP
         */
        struct MPPalm {
            float score{0.0f};
            cv::Rect2f bbox;
            std::array<cv::Point2f, 7> landmarks;

            MPPalm() = default;
            MPPalm(float s, const cv::Rect2f& b, const std::array<cv::Point2f, 7>& l)
             : score(s), bbox(b), landmarks(l) {}
        };

        /// @brief Helper function to extract boxes from output
        inline std::vector<cv::Rect2f> boxes(const std::vector<MPPalm>& detections) {
            std::vector<cv::Rect2f> b(detections.size());
            for ( size_t i = 0; i < detections.size(); i++ )
                b[i] = detections[i].bbox;
            return b;
        }
        
        /**
         * @brief MediaPipe palm detector
         */
        class MPPalmDetector: public core::ModelBase {
            public:
                /// @brief Load model and assert the appropriate shape
                MPPalmDetector(const std::string& model, 
                               const std::string& device = "gpu", 
                               int threads = 0,
                               const std::string& cache_dir = "./trt_cache")
                 : core::ModelBase(model, device, threads, cache_dir), anchors(palm_anchors()) {
                    session_assert_inputs("MPPalmDetector", model, {{1, 192, 192, 3}});
                    session_assert_outputs("MPPalmDetector", model, {
                        {1, 2016, 18},
                        {1, 2016, 1}
                    });
                }

                // Default destructor, no copying
                ~MPPalmDetector() = default;
                MPPalmDetector(const MPPalmDetector&) = delete;
                MPPalmDetector& operator=(const MPPalmDetector&) = delete;

                /// @brief Run inference
                std::vector<MPPalm> detect(const cv::Mat& image,
                                         float score = 0.8f,
                                         float nms = 0.3f) {
                    static const std::vector<int64_t> shape = {1, 192, 192, 3};
                    prepare(image);
                    preprocess(image);
                    Ort::Value tensor = session_tensor(nhwc.data(), shape);
                    std::vector<Ort::Value> results = session_run(tensor);
                    return postprocess(results, score, nms);
                }

                /// @brief Draw a result instance
                void draw(cv::Mat& image, const MPPalm& palm) const {
                    static const cv::Scalar boxcolor(100, 200, 50);
                    static const cv::Scalar pointcolor(50, 200, 0);

                    cv::rectangle(image, cv::Rect(palm.bbox), boxcolor);

                    for ( const auto& l: palm.landmarks ) 
                        cv::circle(image, l, 2, pointcolor);
                }

                /// @brief Draw all detections
                void draw(cv::Mat& image, const std::vector<MPPalm>& palms) const {
                    for ( const auto& p: palms )
                        draw(image, p);
                }

            private:
                cv::Size image_size{0, 0};
                float scale = 0.0f;
                cv::Size roi{0, 0};
                cv::Point2i bias{0, 0};
                cv::Mat blob;
                std::vector<float> nhwc;
                const PalmAnchors& anchors;

                /// @brief Prepare stuff 
                inline void prepare(const cv::Mat& image) {
                    if ( image.cols == image_size.width && image.rows == image_size.height )
                        return;

                    image_size = cv::Size(image.cols, image.rows);
                    scale = std::min(192 / static_cast<float>(image.cols), 192 / static_cast<float>(image.rows));
                    roi = cv::Size(image.cols * scale, image.rows * scale);
                    bias = cv::Point2i((192 - roi.width) / 2, (192 - roi.height) / 2);

                    blob = cv::Mat(192, 192, CV_8UC3);
                    blob.setTo(cv::Scalar(0, 0, 0));

                    nhwc = std::vector<float>(192 * 192 * 3);
                }

                /// @brief Apply preprocessing
                inline void preprocess(const cv::Mat& image) {
                    cv::resize(
                        image,
                        blob(cv::Rect(bias.x, bias.y, roi.width, roi.height)),
                        roi,
                        0, 0, cv::INTER_LINEAR
                    );

                    const uint8_t* src = blob.ptr<uint8_t>();

                    float* dst = nhwc.data();

                    for ( size_t i = 0; i < 192 * 192; i++ ) {
                        dst[3*i + 2] = src[3*i + 0] / 255.0f; // b
                        dst[3*i + 1] = src[3*i + 1] / 255.0f; // g
                        dst[3*i + 0] = src[3*i + 2] / 255.0f; // r
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

                /// @brief Postprocessing
                std::vector<MPPalm> postprocess(const std::vector<Ort::Value>& tensors, float min_score, float nms) {
                    std::vector<cv::Rect2f> boxes;
                    std::vector<float> scores;
                    std::vector<std::array<cv::Point2f, 7>> landmarks;
                    boxes.reserve(64);
                    scores.reserve(64);
                    landmarks.reserve(64);

                    const float* raw_boxes = tensors[0].GetTensorData<float>();
                    const float* raw_scores = tensors[1].GetTensorData<float>();
                    for ( size_t d = 0; d < 2016; d++ ) {
                        float score = sigmoid(raw_scores[d]);

                        if ( score < min_score )
                            continue;

                        scores.push_back(score);

                        const float cxoff = anchors[d].x * 192.0f - bias.x;
                        const float cyoff = anchors[d].y * 192.0f - bias.y;

                        const float cx = raw_boxes[0 + d * 18];
                        const float cy = raw_boxes[1 + d * 18];
                        const float hw = raw_boxes[2 + d * 18] / 2.0f;
                        const float hh = raw_boxes[3 + d * 18] / 2.0f;

                        const float x1 = (cx - hw + cxoff) / scale;
                        const float y1 = (cy - hh + cyoff) / scale;

                        const float x2 = (cx + hw + cxoff) / scale;
                        const float y2 = (cy + hh + cyoff) / scale;

                        boxes.emplace_back(
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1
                        );

                        std::array<cv::Point2f, 7> local;
                        for ( size_t l = 0; l < 7; l++ ) {
                            float x = raw_boxes[4 + l * 2 + d * 18];
                            float y = raw_boxes[5 + l * 2 + d * 18];

                            local[l] = cv::Point2f((x + cxoff) / scale, (y + cyoff) / scale);
                        }
                        landmarks.push_back(local);
                    }

                    std::vector<int> indices;
                    core::nms_boxes(boxes, scores, min_score, nms, indices);

                    std::vector<MPPalm> results;
                    results.reserve(indices.size());
                    for ( int idx: indices )
                        results.emplace_back(scores[idx], boxes[idx], landmarks[idx]);

                    return results;
                }
        };
    }
}

#endif // FOLLOWER_MEDIAPIPE_PALM_HPP_