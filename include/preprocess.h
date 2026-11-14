#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <optional>
#include <filesystem>

/**
 * @file preprocess.h
 * @brief Prepare an image for the model: load, letterbox, convert to a tensor.
 */

/**
 * @brief Load an image file as a BGR, 8-bit, 3-channel OpenCV image.
 * @param imageFile The image to load.
 * @return The image, or nullopt (with an error message) if the file cannot be read.
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
 * @brief Convert a BGR 8-bit image into the model's input layout: planar RGB floats in [0, 1].
 *
 * Writes three planes of rows * cols floats one after the other: all red values, then all green,
 * then all blue (NCHW layout for one image).
 *
 * @param image The image (CV_8UC3, BGR).
 * @param tensorData Buffer of at least 3 * rows * cols floats.
 */
void bgrToPlanarRgb(const cv::Mat& image, float* tensorData);

/**
 * @brief Result of preprocessImage: the input tensor and the letterbox used to build it.
 */
struct PreprocessResult {
    Ort::Value tensor;
    LetterboxInfo letterbox;
};

/**
 * @brief Turn an image into the model's input tensor: letterbox to the model size (see letterbox()),
 * then convert to planar RGB floats (see bgrToPlanarRgb()).
 * @param mat The image (CV_8UC3, BGR).
 * @param inputDims Shape of the model input, [N, C, H, W] with fixed H and W.
 * @param allocator Allocator used to create the tensor.
 * @return The tensor and the letterbox info, or nullopt (with an error message) on invalid input.
 */
std::optional<PreprocessResult> preprocessImage(const cv::Mat& mat,
                                                const std::vector<int64_t>& inputDims,
                                                Ort::AllocatorWithDefaultOptions& allocator);
