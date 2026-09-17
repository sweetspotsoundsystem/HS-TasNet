// Isolated encoder-matrix diagnostic; no audio model, callback, or quality score.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* reason) {
  if (!ok) throw std::runtime_error(reason);
}
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: encoder-kernel MODEL INPUTS OUTPUTS");
    constexpr size_t cases = 16, k = 2048, n = 3000;
    std::vector<float> inputs(cases * k), outputs(cases * n), input(k), output(n);
    std::ifstream stream(argv[2], std::ios::binary);
    require(static_cast<bool>(stream.read(reinterpret_cast<char*>(inputs.data()),
                                         inputs.size() * sizeof(float))), "Read fixture input failed");
    require(stream.peek() == std::ifstream::traits_type::eof(), "Unexpected fixture bytes");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "encoder-kernel");
    Ort::SessionOptions options;
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    Ort::Session session(env, argv[1], options);
    require(session.GetInputCount() == 1 && session.GetOutputCount() == 1,
            "Expected one kernel input and output");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 2> inShape{1, k}, outShape{1, n};
    auto in = Ort::Value::CreateTensor<float>(memory, input.data(), k, inShape.data(), 2);
    auto out = Ort::Value::CreateTensor<float>(memory, output.data(), n, outShape.data(), 2);
    const char* inName = "x";
    const char* outName = "y";
    Ort::RunOptions run;
    for (size_t i = 0; i < cases; ++i) {
      std::copy_n(inputs.data() + i * k, k, input.data());
      session.Run(run, &inName, &in, 1, &outName, &out, 1);
      require(std::all_of(output.begin(), output.end(), [](float v) { return std::isfinite(v); }),
              "Nonfinite kernel output");
      std::copy(output.begin(), output.end(), outputs.begin() + i * n);
    }
    std::ofstream result(argv[3], std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(result.write(reinterpret_cast<const char*>(outputs.data()),
                                          outputs.size() * sizeof(float))), "Write fixture output failed");
    result.close();
    std::vector<double> durations;
    durations.reserve(1024);
    for (size_t i = 0; i < 64 + 1024; ++i) {
      std::copy_n(inputs.data() + (i % cases) * k, k, input.data());
      const auto began = std::chrono::steady_clock::now();
      session.Run(run, &inName, &in, 1, &outName, &out, 1);
      const auto ended = std::chrono::steady_clock::now();
      if (i >= 64) durations.push_back(std::chrono::duration<double, std::micro>(ended - began).count());
    }
    std::sort(durations.begin(), durations.end());
    std::cout.precision(12);
    std::cout << "{\"status\":\"pass\",\"runtime\":\"" << OrtGetApiBase()->GetVersionString()
              << "\",\"inference_threads\":1,\"fixture_cases\":16,\"warmup_calls\":64,\"measured_calls\":1024,"
              << "\"mean_us\":" << std::accumulate(durations.begin(), durations.end(), 0.) / durations.size()
              << ",\"p50_us\":" << (durations[511] + durations[512]) / 2.
              << ",\"p99_us\":" << durations[1013] << ",\"max_us\":" << durations.back()
              << ",\"isolated_kernel_only\":true,\"native_host_qualified\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
