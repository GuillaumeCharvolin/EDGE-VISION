#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <optional>
#include <vector>

#include "preprocess.h"

/**
 * @file postprocess.h
 * @brief Turn the model output tensor into detections on the original image.
 */

/**
 * @brief One detected object.
 */
struct Detection {
    /// Index of the class predicted by the model (COCO index for yolov7-tiny).
    int classId;
    /// Confidence of the detection, in [0, 1].
    float score;
    /// Box as x, y (top-left corner), width, height, in pixels.
    cv::Rect2f box;
};

/**
 * @brief Decode the output of an end-to-end YOLO model (NMS already inside the model).
 *
 * Each row of the output is [batch_id, x0, y0, x1, y1, class_id, score],
 * with coordinates in pixels of the model input (letterboxed) image.
 *
 * @param data Pointer to the float output buffer.
 * @param shape Shape of the output tensor, expected [N, 7].
 * @param minScore Detections with a lower score are dropped.
 * @return The detections in model input coordinates, or nullopt if the shape is wrong.
 */
std::optional<std::vector<Detection>> decodeOutput(const float* data,
                                                   const std::vector<int64_t>& shape,
                                                   float minScore = 0.25f);

/**
 * @brief Map a detection from letterboxed coordinates back to the original image.
 *
 * original = (letterboxed - pad) / scale, then the box is clipped to the image.
 *
 * @param det Detection in letterboxed coordinates.
 * @param info Letterbox parameters used during preprocessing.
 * @param origWidth Width of the original image.
 * @param origHeight Height of the original image.
 * @return The detection in original image coordinates.
 */
Detection unletterbox(const Detection& det, const LetterboxInfo& info, int origWidth, int origHeight);
