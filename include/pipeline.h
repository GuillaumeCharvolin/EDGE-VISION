#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

#include "model.h"
#include "postprocess.h"

/**
 * @file pipeline.h
 * @brief Full detection pipeline on one frame: preprocess, inference, decode, map back.
 */

/**
 * @brief Time spent in each stage for one frame, in milliseconds.
 */
struct StageTimings {
    /// Reading the frame from the camera or video file (capture.read).
    double captureMs = 0.0;
    /// Letterbox + conversion to the input tensor.
    double preprocessMs = 0.0;
    /// Model inference (session.Run).
    double inferenceMs = 0.0;
    /// Decoding the output + mapping boxes back to the frame.
    double postprocessMs = 0.0;
    /// Whole loop iteration: capture to write/display, including drawing (end-to-end).
    double totalMs = 0.0;
};

/**
 * @brief Run the model on one frame and return its detections.
 * @param model A loaded end-to-end YOLO model (NMS inside, output [N, 7]).
 * @param frame The image to analyse (BGR, CV_8UC3).
 * @param timings If not null, its preprocessMs, inferenceMs and postprocessMs are filled.
 * @return The detections in frame coordinates, or nullopt on error (message on stderr).
 */
std::optional<std::vector<Detection>> detect(Model& model, const cv::Mat& frame, StageTimings* timings = nullptr);
