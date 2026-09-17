#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

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
    std::vector<std::string> args = {"edge_vision", "--model", "--image"};
    auto argv = makeArgv(args);
    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST(ArgParserTest, FilesDontExist) {
    std::vector<std::string> args = {
        "edge_vision", "--model", "dontExistModel.onnx", "--image", "dontExistImage.jpg"};
    auto argv = makeArgv(args);
    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

class ArgParserFixture : public ::testing::Test {
protected:
    std::filesystem::path relativeModelPath = "existingModel.onnx";
    std::filesystem::path relativeImagePath = "existingImage.jpg";

    void SetUp() override {
        std::ofstream(relativeModelPath).put('0');
        std::ofstream(relativeImagePath).put('0');
    }

    void TearDown() override {
        std::filesystem::remove(relativeModelPath);
        std::filesystem::remove(relativeImagePath);
    }
};

TEST_F(ArgParserFixture, WrongFormatFiles) {
    // Swap on purpose: image path for --model, model path for --image.
    std::vector<std::string> args = {
        "edge_vision",
        "--model",
        relativeImagePath.string(),
        "--image",
        relativeModelPath.string()};
    auto argv = makeArgv(args);

    EXPECT_EQ(parseArguments(static_cast<int>(args.size()), argv.data()), std::nullopt);
}

TEST_F(ArgParserFixture, CorrectInput) {
    std::vector<std::string> args = {
        "edge_vision",
        "--model",
        relativeModelPath.string(),
        "--image",
        relativeImagePath.string()};
    auto argv = makeArgv(args);

    auto configOpt = parseArguments(static_cast<int>(args.size()), argv.data());

    ASSERT_TRUE(configOpt.has_value()) << "Parser returned nullopt";
    AppConfig config = *configOpt;

    EXPECT_TRUE(config.modelFile.is_absolute());
    EXPECT_TRUE(config.imageFile.is_absolute());

    EXPECT_EQ(config.modelFile, std::filesystem::absolute(relativeModelPath));
    EXPECT_EQ(config.imageFile, std::filesystem::absolute(relativeImagePath));
}
