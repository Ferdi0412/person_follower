#ifndef FOLLOWER_FACE_ARCFACE_HPP_
#define FOLLOWER_FACE_ARCFACE_HPP_

#include "person_follower/core/core.h"

namespace person_follower {
    namespace reid {
        /**
         * @brief Embed faces using the ArcFace model
         */
        class ArcFace: public core::ModelBase {
            public:
                ArcFace(const std::string& model, 
                        const std::string& device="gpu",
                        int threads = 0,
                        const std::string& cache_dir = "")
                 : core::ModelBase(model, device, threads, cache_dir) {
                    session_assert_inputs("ArcFace", model, {{1, 3, 112, 112}});
                    session_assert_outputs("ArcFace", model, {{1, 512}});
                }

                /// @brief Compute embedding from image
                std::vector<float> embed(const cv::Mat& image) {
                    static const std::vector<int64_t> shape = {1, 3, 112, 112};
                    if ( image.type() != CV_8UC3 )
                        throw std::invalid_argument("ArcFace: expects 3-channel BGR");

                    prepare(image);
                    preprocess(image);
                    Ort::Value tensor = session_tensor(bufferb.data(), shape);
                    std::vector<Ort::Value> result = session_run(tensor);
                    return postprocess(result);
                }

                /// @brief Get the required size
                static cv::Size input_size() {
                    return cv::Size(112, 112);
                }

                /// @brief Get the length of embedding
                static size_t output_size() {
                    return 112;
                }

                /// @brief Compute the similarity of two embeddings
                static float similarity(const std::vector<float>& a, const std::vector<float>& b) {
                    float dot = 0.0f;
                    for ( size_t i = 0; i < std::min(a.size(), b.size()); i++ )
                        dot += a[i] * b[i];
                    return dot;
                }

            private:
                cv::Mat buffera;
                std::vector<float> bufferb;

                void prepare(const cv::Mat& image) {
                    cv::cvtColor(image, buffera, cv::COLOR_BGR2RGB);
                    
                    if ( bufferb.size() != 112 * 112 * 3 )
                        bufferb = std::vector<float>(112 * 112 * 3);
                }

                void preprocess(const cv::Mat& image) {
                    for ( size_t v = 0; v < 112; v++ ) {
                        const cv::Vec3b* row = buffera.ptr<cv::Vec3b>(v);
                        for ( size_t u = 0; u < 112; u++ ) {
                            for ( size_t c = 0; c < 3; c++ ) {
                                float val = (static_cast<float>(row[u][c]) - 127.5f) / 128.0f;
                                bufferb[c * 112 * 112 + v * 112 + u] = val;
                            }
                        }
                    }
                }

                std::vector<float> postprocess(const std::vector<Ort::Value>& tensors) {
                    const float* out = tensors[0].GetTensorData<float>();

                    std::vector<float> result(512);

                    float norm = 0.0f;
                    for ( size_t i = 0; i < 512; i++ ) {
                        result[i] = out[i];
                        norm += out[i] * out[i];
                    }

                    norm = std::sqrt(norm);
                    if ( norm > 1e-12f )
                        for ( size_t i = 0; i < 512; i++ )
                            result[i] /= norm;
                    return result;
                }
        };
    }
}

#endif // FOLLOWER_FACE_ARCFACE_HPP_