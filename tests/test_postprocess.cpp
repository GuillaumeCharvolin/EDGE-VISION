#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "postprocess.h"

TEST(DecodeOutputTest, EmptyOutputGivesNoDetection) {
    std::vector<int64_t> shape = {0, 7};
    auto detsOpt = decodeOutput(nullptr, shape);

    ASSERT_TRUE(detsOpt.has_value());
    EXPECT_TRUE(detsOpt->empty());
}

TEST(DecodeOutputTest, RejectsWrongShape) {
    std::vector<float> data(6, 0.0f);
    std::vector<int64_t> shape = {1, 6};

    EXPECT_EQ(decodeOutput(data.data(), shape), std::nullopt);
}

TEST(DecodeOutputTest, DropsLowScoresAndConvertsCornersToWidthHeight) {
    // Rows: [batch_id, x0, y0, x1, y1, class_id, score]
    std::vector<float> data = {
        0, 10, 20, 110, 220, 2, 0.90f,  // kept
        0, 50, 50, 60, 60, 0, 0.10f,    // below minScore
    };
    std::vector<int64_t> shape = {2, 7};

    auto detsOpt = decodeOutput(data.data(), shape, 0.25f);
    ASSERT_TRUE(detsOpt.has_value());
    ASSERT_EQ(detsOpt->size(), 1u);

    const Detection& det = (*detsOpt)[0];
    EXPECT_EQ(det.classId, 2);
    EXPECT_FLOAT_EQ(det.score, 0.90f);
    EXPECT_FLOAT_EQ(det.box.x, 10.0f);
    EXPECT_FLOAT_EQ(det.box.y, 20.0f);
    EXPECT_FLOAT_EQ(det.box.width, 100.0f);
    EXPECT_FLOAT_EQ(det.box.height, 200.0f);
}

TEST(UnletterboxTest, MapsBackToOriginalImage) {
    // 1280x640 image letterboxed into 640x640: scale 0.5, 160 px of padding on top.
    LetterboxInfo info{0.5f, 0, 160};
    Detection det{1, 0.8f, cv::Rect2f(100, 200, 50, 40)};

    Detection out = unletterbox(det, info, 1280, 640);

    EXPECT_EQ(out.classId, 1);
    EXPECT_FLOAT_EQ(out.score, 0.8f);
    EXPECT_FLOAT_EQ(out.box.x, 200.0f);       // 100 / 0.5
    EXPECT_FLOAT_EQ(out.box.y, 80.0f);        // (200 - 160) / 0.5
    EXPECT_FLOAT_EQ(out.box.width, 100.0f);   // 50 / 0.5
    EXPECT_FLOAT_EQ(out.box.height, 80.0f);   // 40 / 0.5
}

TEST(UnletterboxTest, ClipsBoxToImage) {
    // Box starts in the top padding and ends past the right edge.
    LetterboxInfo info{0.5f, 0, 160};
    Detection det{0, 0.5f, cv::Rect2f(600, 150, 100, 60)};

    Detection out = unletterbox(det, info, 1280, 640);

    EXPECT_FLOAT_EQ(out.box.x, 1200.0f);      // 600 / 0.5
    EXPECT_FLOAT_EQ(out.box.y, 0.0f);         // (150 - 160) / 0.5 = -20 → clipped
    EXPECT_FLOAT_EQ(out.box.width, 80.0f);    // right edge 1400 → clipped to 1280
    EXPECT_FLOAT_EQ(out.box.height, 100.0f);  // bottom (210 - 160) / 0.5 = 100
}
