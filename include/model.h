#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

/**
 * @file model.h
 * @brief ONNX Runtime session wrapper (load + metadata)
 */

/**
 * @brief Owns an ORT environment and session for one .onnx file.
 *
 * Member order matters: `session_` must be destroyed before `env_`.
 */
class Model {
public:
    /**
     * @brief Load an ONNX model from disk.
     * @param modelFile Absolute or relative path to a `.onnx` file.
     * @return A ready Model, or nullopt if ORT fails (message on stderr).
     */
    static std::optional<Model> load(const std::filesystem::path& modelFile);

    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Ort::Session& session() { return session_; }
    const Ort::Session& session() const { return session_; }

    const std::vector<std::string>& inputNames() const { return inputNames_; }
    const std::vector<std::string>& outputNames() const { return outputNames_; }

    /** C-string views into inputNames_ / outputNames_ for Ort::Session::Run. */
    const std::vector<const char*>& inputNamePtrs() const { return inputNamePtrs_; }
    const std::vector<const char*>& outputNamePtrs() const { return outputNamePtrs_; }

private:
    Model(Ort::Env env,
          Ort::SessionOptions sessionOptions,
          Ort::Session session,
          std::vector<std::string> inputNames,
          std::vector<std::string> outputNames);

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::Session session_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamePtrs_;
    std::vector<const char*> outputNamePtrs_;
};
