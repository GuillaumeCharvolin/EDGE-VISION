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
 * @brief Run the model on one frame and return its detections.
 * @param model A loaded end-to-end YOLO model (NMS inside, output [N, 7]).
 * @param frame The image to analyse (BGR, CV_8UC3).
 * @return The detections in frame coordinates, or nullopt on error (message on stderr).
 */
std::optional<std::vector<Detection>> detect(Model& model, const cv::Mat& frame);
