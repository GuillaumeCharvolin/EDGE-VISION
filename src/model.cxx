#include "model.h"

#include <iostream>

namespace {

std::vector<std::string> readNodeNames(Ort::Session& session, bool inputs) {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();

    std::vector<std::string> names;
    names.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        Ort::AllocatedStringPtr namePtr =
            inputs ? session.GetInputNameAllocated(i, allocator)
                   : session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(namePtr.get());
    }
    return names;
}

std::vector<const char*> makeCStrPtrs(const std::vector<std::string>& names) {
    std::vector<const char*> ptrs;
    ptrs.reserve(names.size());
    for (const auto& name : names) {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

}  // namespace

Model::Model(Ort::Env env,
             Ort::SessionOptions sessionOptions,
             Ort::Session session,
             std::vector<std::string> inputNames,
             std::vector<std::string> outputNames)
    : env_(std::move(env)),
      sessionOptions_(std::move(sessionOptions)),
      session_(std::move(session)),
      inputNames_(std::move(inputNames)),
      outputNames_(std::move(outputNames)),
      inputNamePtrs_(makeCStrPtrs(inputNames_)),
      outputNamePtrs_(makeCStrPtrs(outputNames_)) {}

std::optional<Model> Model::load(const std::filesystem::path& modelFile) {
    try {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "edge_vision"};
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Linux: path.c_str() is const char*. Windows would need a wide path.
        Ort::Session session{env, modelFile.c_str(), sessionOptions};

        auto inputNames = readNodeNames(session, /*inputs=*/true);
        auto outputNames = readNodeNames(session, /*inputs=*/false);

        if (inputNames.empty() || outputNames.empty()) {
            std::cerr << "ONNX model has no inputs or outputs: " << modelFile
                      << std::endl;
            return std::nullopt;
        }

        return Model{std::move(env),
                     std::move(sessionOptions),
                     std::move(session),
                     std::move(inputNames),
                     std::move(outputNames)};
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error while loading " << modelFile << ": "
                  << e.what() << std::endl;
        return std::nullopt;
    }
}
