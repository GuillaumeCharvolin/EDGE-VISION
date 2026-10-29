#pragma once
#include <ostream>
#include <vector>

#include "postprocess.h"

/**
 * @file serialize.h
 * @brief Write detections to a file format.
 */

/**
 * @brief Write detections as CSV: a header line, then one row per detection.
 *
 * Columns: class_id,class_name,score,x,y,w,h
 * - x, y: top-left corner of the box, w, h: width and height, in original image pixels.
 * - score has 4 decimals, coordinates have 2.
 *
 * @param out Stream to write to (a file, or a string stream in tests).
 * @param detections Detections in original image coordinates.
 */
void writeDetectionsCsv(std::ostream& out, const std::vector<Detection>& detections);
