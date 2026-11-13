#include <gtest/gtest.h>

#include <filesystem>
#include <optional>

#include "pipeline.h"

namespace {

std::filesystem::path fixture(const char* name) {
    return std::filesystem::path{EDGE_VISION_TESTS_DIR} / "fixtures" / name;
}

}  // namespace

TEST(DetectTest, NonImageModelReturnsNullopt) {
    // tiny_add_two.onnx takes a 1D input, so it cannot receive an image tensor.
    std::optional<Model> modelOpt = Model::load(fixture("tiny_add_two.onnx"));
    ASSERT_TRUE(modelOpt.has_value());

    cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(0, 0, 0));
    EXPECT_EQ(detect(*modelOpt, frame), std::nullopt);
}
