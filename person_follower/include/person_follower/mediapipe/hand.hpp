#ifndef PERSON_FOLLOWER_MEDIAPIPE_HAND_HPP_
#define PERSON_FOLLOWER_MEDIAPIPE_HAND_HPP_

#include "person_follower/mediapipe/palm.hpp"

namespace person_follower {
    namespace mediapipe {
        /**
         * @brief Output of MediaPipe hand model
         * @note Landmarks are in order:
         *   0: Wrist, 1-4: Thumb (MCP, PIP, IP, Tip)
         *   5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky (CMC, MCP, PIP, Tip)
         */
        struct MPHand {
            float score{0.0f};
            std::array<cv::Point2f, 21> landmarks;        //< position in image 
            std::array<cv::Point3f, 21> world_landmarks;
            float right_handed{0.0f};

            void offset_inplace(const cv::Point2f& offset) {
                for ( auto& l: landmarks ) {
                    l += offset;
                }
            }

            MPHand apply_offset(const cv::Point2f& offset) {
                MPHand result = *this;
                result.offset_inplace(offset);
                return result;
            }

            // Estimate the palm bounding box
            cv::Rect2f palm_bbox() const {
                constexpr float PAD = 0.15f;
                static constexpr int knuckles[] = {5, 9, 13, 17};

                float min_x = landmarks[0].x;
                float min_y = landmarks[0].y;
                float max_x = min_x;
                float max_y = min_y;

                for ( int i : knuckles ) {
                    const auto& p = landmarks[i];
                    min_x = std::min(p.x, min_x);
                    min_y = std::min(p.y, min_y);
                    max_x = std::max(p.x, max_x);
                    max_y = std::max(p.y, max_y);
                }

                const float w = max_x - min_x;
                const float h = max_y - min_y;
                const float pad_x = w * PAD;
                const float pad_y = w * PAD;

                return cv::Rect2f(min_x - pad_x, min_y - pad_y, w + 2 * pad_x, h + 2 * pad_y);
            }
        };

        /**
         * @brief MediaPipe hand detector
         */
        class MPHandModel: public core::ModelBase {
            public:
                /// @brief Load model and assert the appropriate shape
                MPHandModel(const std::string& model, 
                            const std::string& device = "gpu",
                            int threads = 0,
                            const std::string& cache_dir = "./trt_cache")
                 : core::ModelBase(model, device, threads, cache_dir) {
                    session_assert_inputs("MPHandModel", model, {{1, 224, 224, 3}});
                    session_assert_outputs("MPHandModel", model, {
                        {1, 63},
                        {1, 1},
                        {1, 1},
                        {1, 63}
                    });
                }

                // Default destroyer, no copying
                ~MPHandModel() = default;
                MPHandModel(const MPHandModel&) = delete;
                MPHandModel& operator=(const MPHandModel&) = delete;

                /// @brief Draw result
                void draw(cv::Mat& image, const MPHand& hand) const {
                    static const cv::Scalar boxcolor(100, 200, 50);
                    static const cv::Scalar pointcolor(50, 200, 0);
                    static const cv::Scalar linecolor(50, 200, 0);

                    for ( const auto& l: mp_hand_links() ) {
                        const auto& a = hand.landmarks[l.first];
                        const auto& b = hand.landmarks[l.second];

                        cv::line(image, cv::Point(a.x, a.y), cv::Point(b.x, b.y), linecolor);
                    }
                }
                
                /// @brief Draw with specific color for thumb, index, middle, ring and pinky
                void draw(cv::Mat& image, const MPHand& hand, const std::array<cv::Scalar, 5>& colors) const {
                    draw(image, hand); // Draw links
                    for ( int f = 0; f < 5; f++ ) {
                        for ( int i = 0; i < 4; i++ ) {
                            const auto& p = hand.landmarks[1 + f * 4 + i];
                            cv::circle(image, cv::Point(p), 2, colors[f]);
                        }
                    }
                }

                /// @brief Links to draw
                /// @brief Links to draw
                static const std::array<std::pair<size_t, size_t>, 21>& mp_hand_links() {
                    static const std::array<std::pair<size_t, size_t>, 21> links = {{
                        {0, 1},   {1, 2},   {2, 3},   {3, 4},   //< Thumb
                        {1, 5},   {5, 6},   {6, 7},   {7, 8},   //< Index
                        {5, 9},   {9, 10},  {10, 11}, {11, 12}, //< Middle
                        {9, 13},  {13, 14}, {14, 15}, {15, 16}, //< Ring
                        {13, 17}, {17, 18}, {18, 19}, {19, 20}, //< Pinky
                        {17, 0}
                    }};
                    return links;
                }

                /// @brief Run inference for palm detection
                MPHand detect(const cv::Mat& image, const MPPalm& palm) {
                    cv::RotatedRect roi = compute_alignment(
                        palm.bbox, 
                        palm.landmarks[0], 
                        palm.landmarks[2]
                    );
                    return detect(image, roi);
                }

                /// @brief Run inference for previous hand detection
                MPHand detect(const cv::Mat& image, const MPHand& hand) {
                    cv::RotatedRect roi = compute_alignment(
                        hand.palm_bbox(),
                        hand.landmarks[0],
                        hand.landmarks[5]
                    );
                    MPHand result = detect(image, roi);
                    result.score = result.score * 0.3 + hand.score * 0.7;
                    return result;
                }

            private:
                cv::Size image_size{0, 0};
                cv::Mat buffer;
                std::vector<float> nhwc;

                /// @brief Run detection on desired roi
                MPHand detect(const cv::Mat& image, const cv::RotatedRect& roi) {
                    static const std::vector<int64_t> shape = {1, 224, 224, 3};
                    prepare(image);
                    preprocess(image, roi);
                    Ort::Value tensor = session_tensor(nhwc.data(), shape);
                    std::vector<Ort::Value> results = session_run(tensor);
                    return postprocess(results, roi); 
                }

                /// @brief Prepare stuff
                inline void prepare(const cv::Mat& image) {
                    if ( image.cols == image_size.width && image.rows == image_size.height )
                        return;

                    image_size = cv::Size(image.cols, image.rows);
                    nhwc = std::vector<float>(224 * 224 * 3);
                }

                /// @brief Utility
                inline float normalize_radians(float angle) {
                    constexpr float PI_ = static_cast<float>(M_PI);
                    return angle - 2.0f * PI_ * std::floor((angle + PI_) / (2.0f * PI_));
                }

                /// @brief Preprocessing
                inline void preprocess(const cv::Mat& image, const cv::RotatedRect& roi) {
                    static const cv::Point2f targets[4] = {
                        cv::Point2f(0, 224),
                        cv::Point2f(0, 0),
                        cv::Point2f(224, 0),
                        cv::Point2f(224, 224)
                    };

                    cv::Point2f hand[4];
                    roi.points(hand);

                    cv::Mat affine = cv::getAffineTransform(hand, targets);

                    cv::warpAffine(
                        image,
                        buffer,
                        affine,
                        cv::Size(224, 224),
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT,
                        cv::Scalar(0, 0, 0)
                    );

                    const uint8_t* src = buffer.ptr<uint8_t>();
                    float* dst = nhwc.data();

                    for ( size_t i = 0; i < 224 * 224; i++ ) {
                        dst[3*i + 2] = src[3*i + 0] / 255.0f; // b
                        dst[3*i + 1] = src[3*i + 1] / 255.0f; // g
                        dst[3*i + 0] = src[3*i + 2] / 255.0f; // r
                    }
                }

                /// @brief Postprocessing
                inline MPHand postprocess(const std::vector<Ort::Value>& tensors, const cv::RotatedRect& roi) {
                    MPHand result;

                    result.score = tensors[1].GetTensorData<float>()[0];
                    result.right_handed = tensors[2].GetTensorData<float>()[0];

                    /// @todo Actually compute scale
                    const float scale = roi.size.width;
                    const float ratio = scale / 224.0f;

                    const float angle_rad = roi.angle * static_cast<float>(M_PI) / 180.0f;
                    const float cos_a = std::cos(angle_rad);
                    const float sin_a = std::sin(angle_rad);
                    
                    const float* image_raw = tensors[0].GetTensorData<float>();
                    const float* world_raw = tensors[3].GetTensorData<float>();

                    for ( size_t i = 0; i < 21; i++ ) {
                        const float lx = image_raw[i*3 + 0] - 224.0f / 2.0f;
                        const float ly = image_raw[i*3 + 1] - 224.0f / 2.0f;
                        const float lz = image_raw[i*3 + 2];

                        const float rx = (lx * cos_a - ly * sin_a) * ratio + roi.center.x;
                        const float ry = (lx * sin_a + ly * cos_a) * ratio + roi.center.y;
                        // const float rz = lz * ratio; // no rotation

                        result.landmarks[i] = cv::Point2f(rx, ry);

                        const float wx =  world_raw[i * 3 + 0];
                        const float wy =  world_raw[i * 3 + 1];
                        const float wz =  world_raw[i * 3 + 2];
                        // const float wy = -world_raw[i * 3 + 0];
                        // const float wz = -world_raw[i * 3 + 1]; -> Transform from image to world frame
                        // const float wx =  world_raw[i * 3 + 2]; -> Is already handled by tf

                        result.world_landmarks[i] = cv::Point3f(
                            wx * cos_a - wy * sin_a, //< x
                            wx * sin_a + wy * cos_a, //< y
                            wz  //< z
                        );
                    }

                    return result;
                }

                /// @brief Compute alignment of hand
                inline cv::RotatedRect compute_alignment(const cv::Rect2f& box, 
                                                         const cv::Point2f& wrist, 
                                                         const cv::Point2f& middle_mcp) {
                    constexpr float SCALE_Y = 2.6f;
                    constexpr float SHIFT_Y = -0.5f;

                    constexpr float PI_ = static_cast<float>(M_PI);
                    constexpr float TARGET = PI_ * 0.5f;

                    float cx = box.x + box.width * 0.5f;
                    float cy = box.y + box.height * 0.5f;
                    float width = box.width;
                    float height = box.height;

                    const float rotation = normalize_radians(
                        TARGET - std::atan2(-(middle_mcp.y - wrist.y), middle_mcp.x - wrist.x)
                    );

                    cx -= height * SHIFT_Y * std::sin(rotation);
                    cy += height * SHIFT_Y * std::cos(rotation);

                    width *= SCALE_Y;
                    height *= SCALE_Y;

                    const float long_side = std::max(width, height);

                    width = height = long_side;

                    return cv::RotatedRect(
                        cv::Point2f(cx, cy),
                        cv::Size2f(width, height),
                        rotation * 180.0f / PI_
                    );
                }
        };
    }
}

#endif // FOLLOWER_MEDIAPIPE_HAND_HPP_