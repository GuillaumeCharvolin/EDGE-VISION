#pragma once
#include <ostream>
#include <vector>

#include "pipeline.h"
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

/**
 * @brief Write detections as readable text, for the console: a count line, then one line per detection.
 *
 * Example: "  class=car score=0.758534 x=441.131 y=543.643 w=59.3481 h=47.5384"
 *
 * @param out Stream to write to (std::cout, or a string stream in tests).
 * @param detections Detections in original image coordinates.
 */
void writeDetectionsText(std::ostream& out, const std::vector<Detection>& detections);

/**
 * @brief Write the header line of the timings CSV.
 *
 * Columns: frame,capture_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms
 *
 * @param out Stream to write to.
 */
void writeTimingsCsvHeader(std::ostream& out);

/**
 * @brief Write one row of the timings CSV (one frame), times in milliseconds with 3 decimals.
 *
 * Rows are written one by one, so the file is complete even if the program is stopped with Ctrl+C.
 *
 * @param out Stream to write to.
 * @param frameIndex Index of the frame in the input, starting at 0 (warm-up frames included).
 * @param timings Time spent in each stage for this frame.
 */
void writeTimingsCsvRow(std::ostream& out, int frameIndex, const StageTimings& timings);
