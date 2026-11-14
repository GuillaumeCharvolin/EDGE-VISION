#include <algorithm>
#include <cmath>
#include <iostream>
#include "preprocess.h"

std::optional<cv::Mat> loadImage(const std::filesystem::path& imageFile) {
    cv::Mat image = cv::imread(imageFile.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open the image at: " << imageFile.string() << std::endl;
        return std::nullopt;
    }
    return image;
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

void bgrToPlanarRgb(const cv::Mat& image, float* tensorData) {
    const int planeSize = image.rows * image.cols;
    float* red = tensorData;
    float* green = tensorData + planeSize;
    float* blue = tensorData + 2 * planeSize;

    // OpenCV stores the pixels row by row as B, G, R bytes. The model wants one plane per
    // colour, R first, with values in [0, 1].
    const cv::Mat continuous = image.isContinuous() ? image : image.clone();
    const uint8_t* pixel = continuous.data;
    for (int i = 0; i < planeSize; ++i, pixel += 3) {
        blue[i] = pixel[0] / 255.0f;
        green[i] = pixel[1] / 255.0f;
        red[i] = pixel[2] / 255.0f;
    }
}

std::optional<PreprocessResult> preprocessImage(const cv::Mat& mat,
                                                const std::vector<int64_t>& inputDims,
                                                Ort::AllocatorWithDefaultOptions& allocator) {
    if (inputDims.size() != 4) {
        std::cerr << "Error: Model input must be 4D [N, C, H, W], got " << inputDims.size() << " dimensions."
                  << std::endl;
        return std::nullopt;
    }
    const int64_t targetHeight = inputDims[2];
    const int64_t targetWidth = inputDims[3];
    if (targetWidth <= 0 || targetHeight <= 0) {
        std::cerr << "Error: Model input height and width must be fixed positive sizes." << std::endl;
        return std::nullopt;
    }
    if (mat.type() != CV_8UC3) {
        std::cerr << "Error: Image must be 8-bit with 3 channels (BGR)." << std::endl;
        return std::nullopt;
    }

    try {
        LetterboxInfo info;
        const cv::Mat resized = letterbox(mat, targetWidth, targetHeight, info);

        Ort::Value tensor = Ort::Value::CreateTensor<float>(allocator, inputDims.data(), inputDims.size());
        bgrToPlanarRgb(resized, tensor.GetTensorMutableData<float>());

        return PreprocessResult{std::move(tensor), info};
    } catch (const cv::Exception& e) {
        std::cerr << "Error: OpenCV: " << e.what() << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error: ONNX Runtime: " << e.what() << std::endl;
    }
    return std::nullopt;
}
