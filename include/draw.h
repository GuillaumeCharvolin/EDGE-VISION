#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "postprocess.h"

/**
 * @file draw.h
 * @brief Draw detections on an image.
 */

/**
 * @brief Name of a COCO class (the 80 classes yolov7-tiny was trained on).
 * @param classId Index of the class, in [0, 79].
 * @return The class name, or "unknown" if the index is out of range.
 */
std::string cocoClassName(int classId);

/**
 * @brief Text drawn above a detection box: class name and score with 2 decimals, e.g. "car 0.90".
 */
std::string detectionLabel(const Detection& det);

/**
 * @brief Text of the FPS overlay, with one decimal, e.g. "FPS: 12.3".
 */
std::string fpsLabel(double fps);

/**
 * @brief Draw a rectangle and a "name score" label for each detection, in place.
 * @param mat The image to draw on (BGR, CV_8UC3), in the same coordinates as the boxes.
 * @param detections Detections in image coordinates.
 */
void drawDetections(cv::Mat& mat, const std::vector<Detection>& detections);

/**
 * @brief Draw "FPS: <value>" in the top-left corner of the image, on a black background, in place.
 * @param mat The image to draw on (BGR, CV_8UC3).
 * @param fps Frames per second to display, with one decimal.
 */
void drawFps(cv::Mat& mat, double fps);
