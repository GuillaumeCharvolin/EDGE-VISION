#include <algorithm>
#include <cmath>
#include <iostream>
#include "preprocess.h"

std::optional<cv::Mat> loadImage(const std::filesystem::path& imageFile) {
    cv::Mat img = cv::imread(imageFile.string(), cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "Error: Could not open the image at: " << imageFile.string() << std::endl;
        return std::nullopt;
    }

    return std::optional<cv::Mat>(img);
}

cv::Mat letterbox(const cv::Mat& mat, int targetWidth, int targetHeight, LetterboxInfo& info) {
    info.scale = std::min(static_cast<float>(targetWidth) / mat.cols,
                          static_cast<float>(targetHeight) / mat.rows);

    const int newWidth = static_cast<int>(std::round(mat.cols * info.scale));
    const int newHeight = static_cast<int>(std::round(mat.rows * info.scale));

    // Split the padding in two; the extra pixel (odd padding) goes right / bottom.
    info.padX = (targetWidth - newWidth) / 2;
    info.padY = (targetHeight - newHeight) / 2;

    cv::Mat resized;
    cv::resize(mat, resized, cv::Size(newWidth, newHeight));

    cv::Mat out;
    cv::copyMakeBorder(resized, out,
                       info.padY, targetHeight - newHeight - info.padY,
                       info.padX, targetWidth - newWidth - info.padX,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

std::optional<PreprocessResult> preprocessImage(
    const cv::Mat& mat,
    std::vector<int64_t> input_node_dims,
    Ort::AllocatorWithDefaultOptions& allocator) {

    if (input_node_dims.size() < 4) {
        std::cerr << "Error: Invalid number of parameters for the shaped of aimed tensor." << std::endl;
        return std::nullopt;
    }
    if (mat.channels() != 3) {
        std::cerr << "Input image must have 3 channels (BGR). Got: " << 
                                    std::to_string(mat.channels()) << std::endl;
        return std::nullopt;
    }
    if (mat.depth() != CV_8U) {
        std::cerr << "Input image must be 8-bit unsigned (CV_8U)." << std::endl;
        return std::nullopt;
    }

    int64_t target_height = input_node_dims[2];
    int64_t target_width = input_node_dims[3];

    if (target_width <= 0 || target_height <= 0) {
        std::cerr << "Target dimensions must be greater than zero." << std::endl;
        return std::nullopt;
    }

    try {
        LetterboxInfo info;
        cv::Mat resized = letterbox(mat, target_width, target_height, info);
        // Allocate self-owning tensor via ONNX Runtime*
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            allocator, 
            input_node_dims.data(), 
            input_node_dims.size()
        );

        // Get pointer to the tensor's continuous float buffer
        float* tensor_data = tensor.GetTensorMutableData<float>();
        const int plane_size = target_height * target_width;

        // Planar channel offsets for NCHW [R, G, B]
        float* r_plane = tensor_data;
        float* g_plane = tensor_data + plane_size;
        float* b_plane = tensor_data + (2 * plane_size);

        // Single-pass write: BGR (uint8) -> Planar RGB (float [0.0, 1.0])
        const uint8_t* src_ptr = resized.data;
        const float scale = 1.0f / 255.0f;

        for (int i = 0; i < plane_size; ++i) {
            // OpenCV stores pixels as [B, G, R] interleaved
            b_plane[i] = static_cast<float>(src_ptr[i * 3 + 0]) * scale;
            g_plane[i] = static_cast<float>(src_ptr[i * 3 + 1]) * scale;
            r_plane[i] = static_cast<float>(src_ptr[i * 3 + 2]) * scale;
        }

        return PreprocessResult{std::move(tensor), info};

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return std::nullopt;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return std::nullopt;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory allocating tensor: " << e.what() << std::endl;
        return std::nullopt;
    }
    return std::nullopt;
}
 