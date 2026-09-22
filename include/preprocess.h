#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <optional>
#include <filesystem>

/**
 * @file preprocess.h
 * @brief preprocess image before model inference
 */

/**
 * @brief Transform an image path into an Open CV matrice.
 * @param imageFile The image to open and load
 * @return If loading succeed, a matrice corresponding of the image loader, else nullopt.
 */
std::optional<cv::Mat> loadImage(const std::filesystem::path& imageFile);

/**
 * @brief Transform an Open CV matrice into an ONNX Runtime tensor ready for inference.
 * This function assume that the input image have 3 channels (BGR)
 * and that pixels are 8-bit unsigned (CV_8U).
 * @param mat The matrice corresponding to the image
 * @param input_node_dims the dimension of the required input vector for a specific model.
 * @return If the transformation succed, the tensor ready to use for inference.
 */
std::optional<Ort::Value> preprocessImage(const cv::Mat& mat, std::vector<int64_t> input_node_dims, Ort::AllocatorWithDefaultOptions& allocator);
