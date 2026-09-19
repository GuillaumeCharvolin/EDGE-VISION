#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "model.h"

namespace {

std::filesystem::path fixturesDir() {
    return std::filesystem::path{EDGE_VISION_TESTS_DIR} / "fixtures";
}

std::filesystem::path fixture(const char* name) {
    return fixturesDir() / name;
}

}  // namespace

TEST(ModelTest, LoadMissingFile) {
    ASSERT_EQ(Model::load("dontExistModel.onnx"), std::nullopt);
}

TEST(ModelTest, LoadCorruptFile) {
    ASSERT_EQ(Model::load(fixture("corrupted.onnx")), std::nullopt);
}

TEST(ModelTest, LoadValidModel) {
    std::optional<Model> modelopt = Model::load(fixture("tiny_add_two.onnx"));
    ASSERT_NE(modelopt, std::nullopt);

    Model model = std::move(*modelopt);
    ASSERT_EQ(model.inputNames().size(), 1);
    ASSERT_EQ(model.inputNames()[0], "input");

    ASSERT_EQ(model.outputNames().size(), 1);
    ASSERT_EQ(model.outputNames()[0], "output");

    std::vector<float> input_tensor_values = {5.0f};
    std::vector<int64_t> input_node_dims = {1};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_node_dims.data(), input_node_dims.size());

    auto output_tensors = model.session().Run(
        Ort::RunOptions{nullptr},
        model.inputNamePtrs().data(),
        &input_tensor, 1,
        model.outputNamePtrs().data(), 1);

    ASSERT_EQ(output_tensors.size(), 1);

    const float* out = output_tensors.front().GetTensorData<float>();

    EXPECT_FLOAT_EQ(out[0], 7.0f);
}
