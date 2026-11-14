#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "stream.h"

namespace {

using std::chrono::milliseconds;

std::string readFile(const std::filesystem::path& file) {
    std::ifstream in(file);
    std::stringstream content;
    content << in.rdbuf();
    return content.str();
}

}  // namespace

TEST(FpsMeterTest, FirstFrameGivesInstantFps) {
    const Clock::time_point t0{};
    FpsMeter fps(t0);

    EXPECT_DOUBLE_EQ(fps.tick(t0 + milliseconds(100)), 10.0);  // 1 frame in 0.1 s
}

TEST(FpsMeterTest, LaterFramesAreSmoothed) {
    const Clock::time_point t0{};
    FpsMeter fps(t0);
    fps.tick(t0 + milliseconds(100));  // 10 FPS

    // Next frame 50 ms later = 20 FPS, but only 10% of it counts: 0.9 * 10 + 0.1 * 20.
    EXPECT_DOUBLE_EQ(fps.tick(t0 + milliseconds(150)), 11.0);
}

TEST(FpsMeterTest, MeanFpsIsFramesOverElapsedTime) {
    const Clock::time_point t0{};
    FpsMeter fps(t0);
    fps.tick(t0 + milliseconds(100));
    fps.tick(t0 + milliseconds(150));

    EXPECT_DOUBLE_EQ(fps.meanFps(t0 + milliseconds(200)), 10.0);  // 2 frames in 0.2 s
}

TEST(TimingsLogTest, SkipsWarmupFrames) {
    const std::filesystem::path file = std::filesystem::temp_directory_path() / "edge_vision_timings_test.csv";
    {
        TimingsLog log;
        ASSERT_TRUE(log.open(file, /*warmupFrames=*/2));
        StageTimings timings;
        for (int frame = 0; frame < 4; ++frame) {
            log.record(frame, timings);
        }
    }  // closing the log flushes the file

    EXPECT_EQ(readFile(file),
              "frame,capture_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms\n"
              "2,0.000,0.000,0.000,0.000,0.000\n"
              "3,0.000,0.000,0.000,0.000,0.000\n");
    std::filesystem::remove(file);
}

TEST(TimingsLogTest, RecordDoesNothingWhenNotOpen) {
    TimingsLog log;
    log.record(0, StageTimings{});
    EXPECT_FALSE(log.isOpen());
}

TEST(VideoOutputTest, EmptyPathWritesNothing) {
    VideoOutput video("", 10.0);
    EXPECT_TRUE(video.write(cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 0))));
}

TEST(VideoOutputTest, WritesAllFrames) {
    const std::filesystem::path file = std::filesystem::temp_directory_path() / "edge_vision_video_test.mp4";
    {
        VideoOutput video(file, 10.0);
        for (int i = 0; i < 3; ++i) {
            ASSERT_TRUE(video.write(cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 0, 255))));
        }
    }  // destroying the writer finalises the file

    cv::VideoCapture capture(file.string());
    ASSERT_TRUE(capture.isOpened());
    EXPECT_EQ(capture.get(cv::CAP_PROP_FRAME_COUNT), 3);
    std::filesystem::remove(file);
}
