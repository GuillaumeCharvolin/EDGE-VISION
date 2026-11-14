#include <iostream>
#include "serialize.h"
#include "stream.h"

double millisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

FpsMeter::FpsMeter(Clock::time_point start) : start_(start), previous_(start) {}

double FpsMeter::tick(Clock::time_point now) {
    const double frameSeconds = std::chrono::duration<double>(now - previous_).count();
    previous_ = now;
    const double instantFps = frameSeconds > 0 ? 1.0 / frameSeconds : 0.0;
    smoothedFps_ = frames_ == 0 ? instantFps : 0.9 * smoothedFps_ + 0.1 * instantFps;
    ++frames_;
    return smoothedFps_;
}

double FpsMeter::meanFps(Clock::time_point now) const {
    const double seconds = std::chrono::duration<double>(now - start_).count();
    return seconds > 0 ? frames_ / seconds : 0.0;
}

bool TimingsLog::open(const std::filesystem::path& file, int warmupFrames) {
    csv_.open(file);
    if (!csv_) {
        std::cerr << "Error: Could not open the timings file: " << file.string() << std::endl;
        return false;
    }
    warmupFrames_ = warmupFrames;
    writeTimingsCsvHeader(csv_);
    return true;
}

void TimingsLog::record(int frameIndex, const StageTimings& timings) {
    if (csv_.is_open() && frameIndex >= warmupFrames_) {
        writeTimingsCsvRow(csv_, frameIndex, timings);
    }
}

VideoOutput::VideoOutput(std::filesystem::path file, double fps) : file_(std::move(file)), fps_(fps) {}

bool VideoOutput::write(const cv::Mat& frame) {
    if (file_.empty()) {
        return true;
    }
    if (!writer_.isOpened()) {
        writer_.open(file_.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_, frame.size());
        if (!writer_.isOpened()) {
            std::cerr << "Error: Could not write the output video to: " << file_.string() << std::endl;
            return false;
        }
    }
    writer_.write(frame);
    return true;
}
