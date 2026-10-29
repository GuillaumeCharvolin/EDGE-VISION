#include <algorithm>
#include <iostream>
#include "postprocess.h"

std::optional<std::vector<Detection>> decodeOutput(const float* data,
                                                   const std::vector<int64_t>& shape,
                                                   float minScore) {
    constexpr int64_t rowSize = 7;

    if (shape.size() != 2 || shape[1] != rowSize) {
        std::cerr << "Error: Unexpected model output shape, expected [N, 7]." << std::endl;
        return std::nullopt;
    }

    std::vector<Detection> detections;
    for (int64_t i = 0; i < shape[0]; ++i) {
        // Row layout: [batch_id, x0, y0, x1, y1, class_id, score]
        const float* row = data + i * rowSize;
        const float score = row[6];
        if (score < minScore) {
            continue;
        }

        Detection det;
        det.classId = static_cast<int>(row[5]);
        det.score = score;
        det.box = cv::Rect2f(row[1], row[2], row[3] - row[1], row[4] - row[2]);
        detections.push_back(det);
    }
    return detections;
}

Detection unletterbox(const Detection& det, const LetterboxInfo& info, int origWidth, int origHeight) {
    const float x0 = std::clamp((det.box.x - info.padX) / info.scale, 0.0f, static_cast<float>(origWidth));
    const float y0 = std::clamp((det.box.y - info.padY) / info.scale, 0.0f, static_cast<float>(origHeight));
    const float x1 = std::clamp((det.box.br().x - info.padX) / info.scale, 0.0f, static_cast<float>(origWidth));
    const float y1 = std::clamp((det.box.br().y - info.padY) / info.scale, 0.0f, static_cast<float>(origHeight));

    Detection out = det;
    out.box = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
    return out;
}
