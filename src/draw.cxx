#include <array>
#include <cstdio>
#include "draw.h"

namespace {

const std::array<const char*, 80> cocoNames = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",
    "bus",           "train",        "truck",         "boat",          "traffic light",
    "fire hydrant",  "stop sign",    "parking meter", "bench",         "bird",
    "cat",           "dog",          "horse",         "sheep",         "cow",
    "elephant",      "bear",         "zebra",         "giraffe",       "backpack",
    "umbrella",      "handbag",      "tie",           "suitcase",      "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat",
    "baseball glove", "skateboard",  "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",          "knife",         "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",
    "broccoli",      "carrot",       "hot dog",       "pizza",         "donut",
    "cake",          "chair",        "couch",         "potted plant",  "bed",
    "dining table",  "toilet",       "tv",            "laptop",        "mouse",
    "remote",        "keyboard",     "cell phone",    "microwave",     "oven",
    "toaster",       "sink",         "refrigerator",  "book",          "clock",
    "vase",          "scissors",     "teddy bear",    "hair drier",    "toothbrush",
};

}  // namespace

std::string cocoClassName(int classId) {
    if (classId < 0 || classId >= static_cast<int>(cocoNames.size())) {
        return "unknown";
    }
    return cocoNames[classId];
}

void drawDetections(cv::Mat& mat, const std::vector<Detection>& detections) {
    const cv::Scalar color(0, 255, 0);  // green (BGR)

    for (const Detection& det : detections) {
        cv::rectangle(mat, det.box, color, 2);

        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f", cocoClassName(det.classId).c_str(), det.score);

        // Label just above the box, or inside it if the box touches the top of the image.
        const int textY = det.box.y > 15 ? static_cast<int>(det.box.y) - 5 : static_cast<int>(det.box.y) + 15;
        cv::putText(mat, label, cv::Point(static_cast<int>(det.box.x), textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

void drawFps(cv::Mat& mat, double fps) {
    char label[32];
    std::snprintf(label, sizeof(label), "FPS: %.1f", fps);

    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::rectangle(mat, cv::Rect(0, 0, textSize.width + 20, textSize.height + 20), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(mat, label, cv::Point(10, textSize.height + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
}
