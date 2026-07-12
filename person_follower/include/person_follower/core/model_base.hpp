#ifndef FOLLOWER_CORE_MODEL_BASE_HPP_
#define FOLLOWER_CORE_MODEL_BASE_HPP_

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include "person_follower/core/errors.hpp"
#include "person_follower/core/utils.hpp"

#include <numeric>
#include <string>
#include <vector>

namespace person_follower {
    namespace core {
        /// @brief List devices/providers implemented here (not necessarily available)
        static const std::vector<std::string>& supported_devices() {
            static const std::vector<std::string> providers = {"cpu", "gpu", "cuda", "tensorrt"};
            return providers;
        }

        /**
         * @brief Base for controlling model lifetime
         * This was derived from the approach from YOLOs-CPP
         * See: https://github.com/Geekgineer/YOLOs-CPP
         */
        class ModelBase {
            public:
                /// @brief Immediately loads model
                /// @warning Will crash if device specific backend is not available
                ModelBase(const std::string& model, 
                          const std::string& device, 
                          int threads = 0,
                          const std::string& cache_dir = "./trt_cache")
                 : env(ORT_LOGGING_LEVEL_WARNING, "follower") {
                    set_options(device, threads, cache_dir);
                    init_session(model);
                }

                // Default destructor
                ~ModelBase() = default;

                // Single ownership
                ModelBase(const ModelBase&) = delete;
                ModelBase& operator=(const ModelBase&) = delete;

                /// @brief Run a forward pass
                std::vector<Ort::Value> session_run(const Ort::Value& tensor) {
                    return session.Run(
                        Ort::RunOptions{nullptr},
                        input_names.data(),
                        &tensor,
                        input_names.size(),
                        output_names.data(),
                        output_names.size()
                    );
                }

                /// @brief Create tensor to use for running model
                Ort::Value session_tensor(float* blob, const std::vector<int64_t>& shape) {
                    static Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                    
                    size_t tensor_size = shape.empty() ? 0 : std::accumulate(
                        shape.begin(), 
                        shape.end(), 
                        1ULL, 
                        std::multiplies<size_t>()
                    );

                    return Ort::Value::CreateTensor<float>(
                        memory_info, 
                        blob, 
                        tensor_size, 
                        shape.data(), 
                        shape.size()
                    );
                }

                /// @brief Number of input nodes 
                size_t session_input_count() const {
                    return n_inputs;
                }

                /// @brief Number of output nodes
                size_t session_output_count() const {
                    return n_outputs;
                }

                /// @brief Get shape of a session input
                std::vector<int64_t> session_input(size_t node = 0) const {
                    Ort::TypeInfo type_info = session.GetInputTypeInfo(node);
                    return type_info.GetTensorTypeAndShapeInfo().GetShape();
                }

                /// @brief Get shape of a session output
                std::vector<int64_t> session_output(size_t node = 0) const {
                    Ort::TypeInfo type_info = session.GetOutputTypeInfo(node);
                    return type_info.GetTensorTypeAndShapeInfo().GetShape();
                }

                /// @brief Formatting for useful error messages
                static inline std::string format_shape(const std::vector<int64_t>& shape) {
                    std::string str = "{ ";
                    for ( int64_t s: shape )
                        str += std::to_string(s) + ", ";
                    return str + "}";
                }

                /// @brief Throw InvalidModel if input shape is unexpected 
                void session_assert_inputs(const std::string& cls, 
                                           const std::string& model, 
                                           const std::vector<std::vector<int64_t>>& expected) const {
                    if ( expected.size() != session_input_count() )
                        throw InvalidModel(
                            cls, 
                            model,
                            "Expected " + std::to_string(expected.size()) + 
                            " inputs, got " + std::to_string(session_input_count())
                        );

                    for ( size_t i = 0; i < expected.size(); i++ ) {
                        std::vector<int64_t> actual = session_input(i);
                        if ( actual.size() != expected[i].size() )
                            throw InvalidModel(
                                cls,
                                model,
                                "Expected input " + std::to_string(i) +
                                " to have size " + std::to_string(expected.size()) +
                                " but had " + std::to_string(actual.size())
                            );
                        for ( size_t j = 0; j < actual.size(); j++ )
                            if ( actual[j] != expected[i][j] )
                                throw InvalidModel(
                                    cls,
                                    model,
                                    "Expected input (" + std::to_string(i) + 
                                    ") to be " + format_shape(expected[i]) +
                                    " but got " + format_shape(actual)
                                );
                    }
                }

                /// @brief Throw InvalidModel if output shape is unexpected 
                void session_assert_outputs(const std::string& cls, 
                                            const std::string& model, 
                                            const std::vector<std::vector<int64_t>>& expected) const {
                    if ( expected.size() != session_output_count() )
                        throw InvalidModel(
                            cls, 
                            model,
                            "Expected " + std::to_string(expected.size()) + 
                            " outputs, got " + std::to_string(session_output_count())
                        );

                    for ( size_t i = 0; i < expected.size(); i++ ) {
                        std::vector<int64_t> actual = session_output(i);
                        if ( actual.size() != expected[i].size() )
                            throw InvalidModel(
                                cls,
                                model,
                                "Expected output " + std::to_string(i) +
                                " to have size " + std::to_string(expected.size()) +
                                " but had " + std::to_string(actual.size())
                            );
                        for ( size_t j = 0; j < actual.size(); j++ )
                            if ( actual[j] != expected[i][j] )
                                throw InvalidModel(
                                    cls,
                                    model,
                                    "Expected output (" + std::to_string(i) + 
                                    ") to be " + format_shape(expected[i]) +
                                    " but got " + format_shape(actual)
                                );
                    }
                }

            private:
                void set_options(const std::string& device, int threads = 0, const std::string& cache_dir = "") {
                    options = Ort::SessionOptions();
                    
                    if ( threads > 0 )
                        options.SetIntraOpNumThreads(threads);

                    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

                    if ( device == "gpu" || device == "cuda" ) {
                        OrtCUDAProviderOptions cudaopts{};
                        options.AppendExecutionProvider_CUDA(cudaopts);
                    }
                    else if ( device == "tensorrt" ) {
                        OrtTensorRTProviderOptions trtopts{};

                        // Defaults
                        trtopts.trt_max_partition_iterations = 1000;
                        trtopts.trt_min_subgraph_size = 10;
                        // Max 256 MB
                        trtopts.trt_max_workspace_size = 256 * (1024 * 1024);
                        // trtopts.trt_context_memory_sharing_enable = 1; // Not available 

                        // Cache TensorRT for quicker subsequent startup
                        if ( cache_dir.size() ) {
                            trtopts.trt_engine_cache_enable = 1;
                            trtopts.trt_engine_cache_path = cache_dir.c_str();
                            trtopts.trt_fp16_enable = 0; // 1;
                        }

                        options.AppendExecutionProvider_TensorRT(trtopts);
                    }
                    // else "cpu"
                }

                void init_session(const std::string& model) {
                    session = Ort::Session(env, model.c_str(), options);

                    Ort::AllocatorWithDefaultOptions allocator;

                    n_inputs = session.GetInputCount();
                    for ( size_t i = 0; i < n_inputs; i++ ) {
                        input_name_allocs.push_back(session.GetInputNameAllocated(i, allocator));
                        input_names.push_back(input_name_allocs.back().get());
                    }

                    n_outputs = session.GetOutputCount();
                    for ( size_t i = 0; i < n_outputs; i++ ) {
                        output_name_allocs.push_back(session.GetOutputNameAllocated(i, allocator));
                        output_names.push_back(output_name_allocs.back().get());
                    }
                }

            private:
                size_t n_inputs;
                size_t n_outputs;

                std::vector<Ort::AllocatedStringPtr> input_name_allocs;
                std::vector<const char*> input_names;
                std::vector<Ort::AllocatedStringPtr> output_name_allocs;
                std::vector<const char*> output_names;

                Ort::Env env{nullptr};
                Ort::SessionOptions options{nullptr};
                Ort::Session session{nullptr};
        };
    }
}

#endif // FOLLOWER_CORE_MODEL_BASE_HPP_