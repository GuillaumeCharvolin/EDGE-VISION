#pragma once
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
 * @brief Parameters of a letterbox transform, needed to map boxes back to the original image.
 *
 * original = (letterboxed - pad) / scale
 */
struct LetterboxInfo {
    /// Factor applied to the original image (same on both axes to keep aspect ratio).
    float scale;
    /// Padding added on the left, in pixels of the letterboxed image.
    int padX;
    /// Padding added on the top, in pixels of the letterboxed image.
    int padY;
};

/**
 * @brief Resize an image to targetWidth x targetHeight while keeping its aspect ratio,
 * filling the remaining border with gray (114, 114, 114).
 * @param mat The source image.
 * @param targetWidth Width of the output image.
 * @param targetHeight Height of the output image.
 * @param info Filled with the scale and padding used.
 * @return The letterboxed image, of size targetWidth x targetHeight.
 */
cv::Mat letterbox(const cv::Mat& mat, int targetWidth, int targetHeight, LetterboxInfo& info);

/**
 * @brief Result of preprocessImage: the input tensor and the letterbox used to build it.
 */
struct PreprocessResult {
    Ort::Value tensor;
    LetterboxInfo letterbox;
};

/**
 * @brief Transform an Open CV matrice into an ONNX Runtime tensor ready for inference.
 * The image is letterboxed to the model input size (see letterbox()).
 * This function assume that the input image have 3 channels (BGR)
 * and that pixels are 8-bit unsigned (CV_8U).
 * @param mat The matrice corresponding to the image
 * @param input_node_dims the dimension of the required input vector for a specific model.
 * @return If the transformation succed, the tensor ready to use for inference and the letterbox info.
 */
std::optional<PreprocessResult> preprocessImage(const cv::Mat& mat, std::vector<int64_t> input_node_dims, Ort::AllocatorWithDefaultOptions& allocator);
