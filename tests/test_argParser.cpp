#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>  // getpid

#include "argParser.h"
#include "appConfig.h"

namespace {

std::vector<char*> makeArgv(std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& s : args) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);
    return argv;
}

}

TEST(ArgParserTest, NoArguments) {
    std::vector<std::string> args = {"edge_vision"};
    auto argv = makeArgv(args);
    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST(ArgParserTest, UnknownArguments) {
    std::vector<std::string> args = {"edge_vision", "--wrong"};
    auto argv = makeArgv(args);
    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST(ArgParserTest, ArgumentsWithoutFiles) {
    std::vector<std::string> args = {"edge_vision", "--model", "--input"};
    auto argv = makeArgv(args);
    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST(ArgParserTest, FilesDontExist) {
    std::vector<std::string> args = {
        "edge_vision", "--model", "dontExistModel.onnx", "--input", "dontExistImage.jpg"};
    auto argv = makeArgv(args);
    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

// Each test runs in its own new, empty folder, with three small files to point the options at.
// The paths stay relative, so the tests can check that the parser makes them absolute.
// A separate folder per test matters when tests run in parallel (ctest -j): with one shared
// folder, a test finishing deletes the files another test is still using.
class ArgParserFixture : public ::testing::Test {
protected:
    std::filesystem::path relativeModelPath = "existingModel.onnx";
    std::filesystem::path relativeImagePath = "existingImage.jpg";
    std::filesystem::path relativeVideoPath = "existingVideo.avi";

    void SetUp() override {
        const std::string testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        testDir_ = std::filesystem::temp_directory_path() /
                   ("edge_vision_argParser_" + testName + "_" + std::to_string(::getpid()));
        std::filesystem::create_directories(testDir_);
        previousDir_ = std::filesystem::current_path();
        std::filesystem::current_path(testDir_);

        std::ofstream(relativeModelPath).put('0');
        std::ofstream(relativeImagePath).put('0');
        std::ofstream(relativeVideoPath).put('0');
    }

    void TearDown() override {
        std::filesystem::current_path(previousDir_);
        std::filesystem::remove_all(testDir_);
    }

private:
    std::filesystem::path testDir_;
    std::filesystem::path previousDir_;
};

TEST_F(ArgParserFixture, WrongFormatFiles) {
    // Swap on purpose: image path for --model, model path for --input.
    std::vector<std::string> args = {
        "edge_vision",
        "--model",
        relativeImagePath.string(),
        "--input",
        relativeModelPath.string()};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, CorrectInput) {
    std::vector<std::string> args = {
        "edge_vision",
        "--model",
        relativeModelPath.string(),
        "--input",
        relativeImagePath.string()};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());

    ASSERT_TRUE(configOpt.has_value()) << "Parser returned nullopt";
    AppConfig config = *configOpt;

    EXPECT_TRUE(config.modelFile.is_absolute());
    EXPECT_TRUE(config.inputFile.is_absolute());

    EXPECT_EQ(config.modelFile, std::filesystem::absolute(relativeModelPath));
    EXPECT_EQ(config.inputFile, std::filesystem::absolute(relativeImagePath));
}

TEST_F(ArgParserFixture, OutputIsOptional) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string()};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    EXPECT_TRUE(configOpt->outputFile.empty());
}

TEST_F(ArgParserFixture, OutputMustBeJpg) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--output", "out.png"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, OutputDirectoryMustExist) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--output", "dontExistDir/out.jpg"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, OutputIsMadeAbsolute) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--output", "out.jpg"};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    EXPECT_TRUE(configOpt->outputFile.is_absolute());
    EXPECT_EQ(configOpt->outputFile, std::filesystem::absolute("out.jpg"));
}

TEST_F(ArgParserFixture, DetectionsAcceptsCsv) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--detections", "dets.csv"};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    EXPECT_EQ(configOpt->detectionsFile, std::filesystem::absolute("dets.csv"));
}

TEST_F(ArgParserFixture, DetectionsMustBeCsv) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--detections", "dets.txt"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, DetectionsDirectoryMustExist) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--detections", "dontExistDir/dets.csv"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST(IsVideoFileTest, KnowsVideoExtensions) {
    EXPECT_TRUE(isVideoFile("clip.mp4"));
    EXPECT_TRUE(isVideoFile("dir/clip.avi"));
    EXPECT_FALSE(isVideoFile("image.jpg"));
    EXPECT_FALSE(isVideoFile("clip"));
}

TEST_F(ArgParserFixture, VideoInputAccepted) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeVideoPath.string(),
        "--output", "out.mp4"};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    EXPECT_EQ(configOpt->inputFile, std::filesystem::absolute(relativeVideoPath));
    EXPECT_EQ(configOpt->outputFile, std::filesystem::absolute("out.mp4"));
}

TEST_F(ArgParserFixture, VideoInputNeedsMp4Output) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeVideoPath.string(),
        "--output", "out.jpg"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, ImageInputNeedsJpgOutput) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--output", "out.mp4"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, VideoInputRefusesDetections) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeVideoPath.string(),
        "--detections", "dets.csv"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, CameraIndexAccepted) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", "0", "--output", "out.mp4"};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    ASSERT_TRUE(configOpt->cameraIndex.has_value());
    EXPECT_EQ(*configOpt->cameraIndex, 0);
    EXPECT_TRUE(configOpt->inputFile.empty());
}

TEST_F(ArgParserFixture, FileInputHasNoCameraIndex) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string()};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    EXPECT_FALSE(configOpt->cameraIndex.has_value());
}

TEST_F(ArgParserFixture, NegativeCameraIndexRejected) {
    std::vector<std::string> args = {"edge_vision", "--model", relativeModelPath.string(), "--input", "-1"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, CameraNeedsMp4Output) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", "0", "--output", "out.jpg"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, CameraRefusesDetections) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", "0", "--detections", "dets.csv"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, TimingsWithVideoAccepted) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeVideoPath.string(),
        "--timings", "timings.csv"};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    EXPECT_EQ(configOpt->timingsFile, std::filesystem::absolute("timings.csv"));
    EXPECT_EQ(configOpt->warmupFrames, 10);
}

TEST_F(ArgParserFixture, TimingsWithImageRejected) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeImagePath.string(),
        "--timings", "timings.csv"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, WarmupIsSet) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeVideoPath.string(),
        "--warmup", "0"};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());
    ASSERT_TRUE(configOpt.has_value());
    EXPECT_EQ(configOpt->warmupFrames, 0);
}

TEST_F(ArgParserFixture, NegativeWarmupRejected) {
    std::vector<std::string> args = {
        "edge_vision", "--model", relativeModelPath.string(), "--input", relativeVideoPath.string(),
        "--warmup", "-1"};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}
