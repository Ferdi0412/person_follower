#ifndef FOLLOWER_REID_OSNET_HPP_
#define FOLLOWER_REID_OSNET_HPP_

#include "person_follower/core/core.h"

namespace person_follower {
    namespace reid {
        inline float similarity(const std::vector<float>& a, const std::vector<float>& b) {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                dot    += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
            return dot / std::max(denom, 1e-6f);
        }

        /**
         * @brief OSNet for producing embeddings of people for re-id
         */
        class OSNet: public core::ModelBase {
            public:
                /// @brief Load model and assert the appropriate shapes
                OSNet(const std::string& model,
                      const std::string& device = "gpu",
                      int threads = 0,
                      const std::string& cache_dir = "./trt_cache")
                 : core::ModelBase(model, device, threads, cache_dir) {
                    session_assert_inputs("OSNet", model, {{1, 3, 256, 128}});
                    session_assert_outputs("OSNet", model, {{1, 512}});
                }

                // Default destructor, no copying
                ~OSNet() = default;
                OSNet(const OSNet&) = delete;
                OSNet& operator=(const OSNet&) = delete;

                /// @brief Embed input
                /// Expects the input to be appropriately cropped
                /// Otherwise just applies resize with stretch
                std::vector<float> embed(const cv::Mat& image) {
                    static const std::vector<int64_t> shape = {1, 3, 256, 128};
                    preprocess(image);
                    Ort::Value tensor = session_tensor(chw.data(), shape);
                    std::vector<Ort::Value> results = session_run(tensor);
                    return postprocess(results);
                }

            private:
                std::array<float, 128 * 256 * 3> chw;

                void preprocess(const cv::Mat& image) {
                    cv::Mat resized;

                    if ( image.cols != 128 || image.rows != 256 )
                        cv::resize(image, resized, cv::Size(128, 256), 0, 0, cv::INTER_LINEAR);
                    else
                        resized = image;

                    const uint8_t* src = resized.ptr<uint8_t>();

                    float* r = chw.data();
                    float* g = chw.data() + 1 * 128 * 256;
                    float* b = chw.data() + 2 * 128 * 256;

                    for ( size_t i = 0; i < 128 * 256; i++ ) {
                        b[i] = (src[3*i + 0] / 255.0f - 0.406f) / 0.225f;
                        g[i] = (src[3*i + 1] / 255.0f - 0.456f) / 0.224f;
                        r[i] = (src[3*i + 2] / 255.0f - 0.485f) / 0.229f; 
                    }
                }


                std::vector<float> postprocess(const std::vector<Ort::Value>& tensors) {
                    std::vector<float> result(512);
                    
                    const float* data = tensors[0].GetTensorData<float>();
                    float norm = 0.0f;

                    for ( size_t i = 0; i < 512; i++ )
                        norm += data[i] * data[i];
                    
                    norm = std::sqrt(std::max(norm, static_cast<float>(1e-12)));

                    for ( size_t i = 0; i < 512; i++ ) 
                        result[i] = data[i] / norm;

                    return result;
                }
        };
    }
}

#endif // FOLLOWER_REID_OSNET_HPP_