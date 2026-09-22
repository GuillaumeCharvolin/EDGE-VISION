#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include "preprocess.h"

namespace {

std::filesystem::path fixturesDir() {
    return std::filesystem::path{EDGE_VISION_TESTS_DIR} / "fixtures";
}

std::filesystem::path fixture(const char* name) {
    return fixturesDir() / name;
}

bool nearlyEqual(float a, float b, float eps = 1e-2f) {
    return std::fabs(a - b) <= eps;
}

}  // namespace

TEST(LoadImageTest, MissingFileReturnsNullopt) {
    EXPECT_EQ(loadImage("does_not_exist.jpg"), std::nullopt);
}

TEST(LoadImageTest, CorruptFileReturnsNullopt) {
    EXPECT_EQ(loadImage(fixture("corrupted.jpg")), std::nullopt);
}

TEST(LoadImageTest, LoadsColorJpeg) {
    auto matOpt = loadImage(fixture("gray_8x8.jpg"));
    ASSERT_TRUE(matOpt.has_value());

    const cv::Mat& mat = *matOpt;
    EXPECT_FALSE(mat.empty());
    EXPECT_EQ(mat.rows, 8);
    EXPECT_EQ(mat.cols, 8);
    EXPECT_EQ(mat.channels(), 3);
    EXPECT_EQ(mat.depth(), CV_8U);
}

TEST(PreprocessImageTest, RejectsBadRank) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 255));  // BGR red
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> dims = {1, 3, 2};  // not 4D

    EXPECT_EQ(preprocessImage(mat, dims, allocator), std::nullopt);
}

TEST(PreprocessImageTest, RejectsWrongChannelCount) {
    cv::Mat gray(2, 2, CV_8UC1, cv::Scalar(128));
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> dims = {1, 3, 2, 2};

    EXPECT_EQ(preprocessImage(gray, dims, allocator), std::nullopt);
}

TEST(PreprocessImageTest, RejectsDynamicOrNonPositiveSpatialDims) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 255));
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> dims = {1, 3, -1, 2};

    EXPECT_EQ(preprocessImage(mat, dims, allocator), std::nullopt);
}

TEST(PreprocessImageTest, RedImageFillsRPlaneInNchw) {
    auto matOpt = loadImage(fixture("solid_red_2x2.jpg"));
    ASSERT_TRUE(matOpt.has_value());

    Ort::AllocatorWithDefaultOptions allocator;
    // NCHW: N=1, C=3, H=2, W=2
    std::vector<int64_t> dims = {1, 3, 2, 2};

    auto tensorOpt = preprocessImage(*matOpt, dims, allocator);
    ASSERT_TRUE(tensorOpt.has_value());

    Ort::Value& tensor = *tensorOpt;
    ASSERT_TRUE(tensor.IsTensor());

    auto info = tensor.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(info.GetElementCount(), 1 * 3 * 2 * 2);

    const float* data = tensor.GetTensorData<float>();
    const int plane = 2 * 2;

    for (int i = 0; i < plane; ++i) {
        EXPECT_TRUE(nearlyEqual(data[i], 1.0f)) << "R plane index " << i;
        EXPECT_TRUE(nearlyEqual(data[plane + i], 0.0f)) << "G plane index " << i;
        EXPECT_TRUE(nearlyEqual(data[2 * plane + i], 0.0f)) << "B plane index " << i;
    }
}

TEST(PreprocessImageTest, UsesHeightWidthFromNchwLayout) {
    auto matOpt = loadImage(fixture("solid_blue_4x6.jpg"));
    ASSERT_TRUE(matOpt.has_value());
    EXPECT_EQ(matOpt->rows, 4);
    EXPECT_EQ(matOpt->cols, 6);

    Ort::AllocatorWithDefaultOptions allocator;
    // Target H=4, W=8 (wider than source) — must use dims[2]=H, dims[3]=W
    std::vector<int64_t> dims = {1, 3, 4, 8};

    auto tensorOpt = preprocessImage(*matOpt, dims, allocator);
    ASSERT_TRUE(tensorOpt.has_value());

    auto shape = tensorOpt->GetTensorTypeAndShapeInfo().GetShape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);  // H
    EXPECT_EQ(shape[3], 8);  // W

    const float* data = tensorOpt->GetTensorData<float>();
    const int plane = 4 * 8;
    // Pure blue RGB → R=0, G=0, B=1 after /255
    for (int i = 0; i < plane; ++i) {
        EXPECT_TRUE(nearlyEqual(data[i], 0.0f)) << "R " << i;
        EXPECT_TRUE(nearlyEqual(data[plane + i], 0.0f)) << "G " << i;
        EXPECT_TRUE(nearlyEqual(data[2 * plane + i], 1.0f)) << "B " << i;
    }
}
