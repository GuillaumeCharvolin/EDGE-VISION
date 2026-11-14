#include <iomanip>
#include "draw.h"
#include "serialize.h"

void writeDetectionsCsv(std::ostream& out, const std::vector<Detection>& detections) {
    out << "class_id,class_name,score,x,y,w,h\n";

    for (const Detection& det : detections) {
        out << det.classId << ',' << cocoClassName(det.classId) << ','
            << std::fixed << std::setprecision(4) << det.score << ','
            << std::setprecision(2)
            << det.box.x << ',' << det.box.y << ','
            << det.box.width << ',' << det.box.height << '\n';
    }
}

void writeTimingsCsvHeader(std::ostream& out) {
    out << "frame,capture_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms\n";
}

void writeTimingsCsvRow(std::ostream& out, int frameIndex, const StageTimings& timings) {
    out << frameIndex << ',' << std::fixed << std::setprecision(3)
        << timings.captureMs << ',' << timings.preprocessMs << ','
        << timings.inferenceMs << ',' << timings.postprocessMs << ','
        << timings.totalMs << '\n';
}
