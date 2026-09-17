// Standalone CPU1 timing probe for the hop128 ABI. This is not a DAW test.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::array<const char*, 5> inputs{
    "audio_chunk", "audio_history", "fusion_hidden",
    "spectral_numerator_tail", "waveform_tail"};
constexpr std::array<const char*, 5> outputs{
    "separated_chunk", "next_audio_history", "next_fusion_hidden",
    "next_spectral_numerator_tail", "next_waveform_tail"};
const std::array<std::vector<int64_t>, 5> inputShapes{
    std::vector<int64_t>{1, 2, 128}, {1, 2, 896}, {2, 1, 1000},
    {1, 4, 2, 128}, {1, 4, 2, 128}};
const std::array<std::vector<int64_t>, 5> outputShapes{
    std::vector<int64_t>{1, 4, 2, 128}, {1, 2, 896}, {2, 1, 1000},
    {1, 4, 2, 128}, {1, 4, 2, 128}};

void require(bool condition, const char* reason) {
  if (!condition) throw std::runtime_error(reason);
}

size_t elements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<size_t>{});
}

double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

void statistics(std::vector<double> values) {
  const double mean = std::accumulate(values.begin(), values.end(), 0.) /
                      static_cast<double>(values.size());
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double fraction) {
    const double index = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<size_t>(index);
    const auto upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (index - static_cast<double>(lower)) *
                               (values[upper] - values[lower]);
  };
  constexpr double budget = 128000. / 44100.;
  std::cout << "{\"mean_ms\":" << mean << ",\"p50_ms\":" << percentile(.5)
            << ",\"p95_ms\":" << percentile(.95)
            << ",\"p99_ms\":" << percentile(.99)
            << ",\"maximum_ms\":" << values.back()
            << ",\"hop_budget_ms\":" << budget
            << ",\"calls_over_hop_budget\":"
            << std::count_if(values.begin(), values.end(),
                             [](double value) { return value > budget; })
            << "}";
}

void disableDenormals() {
#if defined(__aarch64__)
  uint64_t fpcr = 0;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  fpcr |= uint64_t{1} << 24;
  asm volatile("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__SSE2__)
  _mm_setcsr(_mm_getcsr() | 0x8040U);
#else
  throw std::runtime_error("Unsupported denormal-control architecture");
#endif
}

const char* configurePriority() {
#if defined(__APPLE__)
  require(pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0,
          "Cannot apply the plugin's Mac worker QoS");
  qos_class_t observed{};
  int relative = 0;
  require(pthread_get_qos_class_np(pthread_self(), &observed, &relative) == 0 &&
              observed == QOS_CLASS_USER_INTERACTIVE,
          "Mac worker QoS readback failed");
  return "QOS_CLASS_USER_INTERACTIVE (set and read back)";
#else
  return "OS default; Linux inference diagnostic";
#endif
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: benchmark MODEL.onnx WARMUP_HOPS MEASURED_HOPS");
    const int warmup = std::stoi(argv[2]);
    const int measured = std::stoi(argv[3]);
    require(warmup >= 16 && warmup <= 16384 && measured >= 64 && measured <= 65536,
            "Hop counts outside bounded range");
    disableDenormals();
    const char* priority = configurePriority();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "latency58-native-probe");
    Ort::SessionOptions options;
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    Ort::Session session(env, argv[1], options);  // Default CPU EP only.
    Ort::AllocatorWithDefaultOptions allocator;
    require(session.GetInputCount() == 5 && session.GetOutputCount() == 5,
            "Expected five inputs and outputs");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<std::vector<float>, 5> in, out;
    std::vector<Ort::Value> inputValues, outputValues;
    for (size_t i = 0; i < 5; ++i) {
      auto inputName = session.GetInputNameAllocated(i, allocator);
      auto outputName = session.GetOutputNameAllocated(i, allocator);
      require(std::string(inputName.get()) == inputs[i] &&
                  std::string(outputName.get()) == outputs[i], "Tensor names changed");
      auto inputType = session.GetInputTypeInfo(i);
      auto outputType = session.GetOutputTypeInfo(i);
      auto inputInfo = inputType.GetTensorTypeAndShapeInfo();
      auto outputInfo = outputType.GetTensorTypeAndShapeInfo();
      require(inputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                  outputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                  inputInfo.GetShape() == inputShapes[i] &&
                  outputInfo.GetShape() == outputShapes[i], "Static FP32 ABI changed");
      in[i].resize(elements(inputShapes[i]), 0.f);
      out[i].resize(elements(outputShapes[i]), 0.f);
      inputValues.push_back(Ort::Value::CreateTensor<float>(
          memory, in[i].data(), in[i].size(), inputShapes[i].data(), inputShapes[i].size()));
      outputValues.push_back(Ort::Value::CreateTensor<float>(
          memory, out[i].data(), out[i].size(), outputShapes[i].data(), outputShapes[i].size()));
    }
    // Deterministic stereo broadband/tone input; generation is outside timing.
    // Synthetic data measures runtime only, never separation quality.
    std::vector<float> audio(static_cast<size_t>(warmup + measured) * 256);
    uint32_t random = 20260912U;
    for (int hop = 0; hop < warmup + measured; ++hop) {
      for (int channel = 0; channel < 2; ++channel) {
        for (int sample = 0; sample < 128; ++sample) {
          random = 1664525U * random + 1013904223U;
          const double noise = static_cast<double>(random) / 4294967296. - .5;
          const double t = static_cast<double>(hop * 128 + sample) / 44100.;
          audio[static_cast<size_t>(hop) * 256 + channel * 128 + sample] =
              static_cast<float>(.07 * noise + .12 * std::sin((220. + 31. * channel) *
                                                           6.283185307179586 * t));
        }
      }
    }
    std::array<float, 256> previous{};
    std::vector<double> runTimes(static_cast<size_t>(measured));
    std::vector<double> loopTimes(static_cast<size_t>(measured));
    Ort::RunOptions runOptions;
    double maximumClosure = 0.;
    for (int hop = 0; hop < warmup + measured; ++hop) {
      const auto began = Clock::now();
      std::copy_n(audio.data() + static_cast<size_t>(hop) * 256, 256, in[0].data());
      const auto runBegan = Clock::now();
      session.Run(runOptions, inputs.data(), inputValues.data(), 5,
                  outputs.data(), outputValues.data(), 5);
      const auto runEnd = Clock::now();
      for (size_t i = 0; i < 5; ++i) {
        for (float value : out[i]) require(std::isfinite(value), "Nonfinite output/state");
        if (i != 0) std::copy(out[i].begin(), out[i].end(), in[i].begin());
      }
      for (size_t sample = 0; sample < 256; ++sample) {
        const float sum = out[0][sample] + out[0][256 + sample] +
                          out[0][512 + sample] + out[0][768 + sample];
        maximumClosure = std::max(maximumClosure,
                                 std::abs(static_cast<double>(sum - previous[sample])));
      }
      std::copy(in[0].begin(), in[0].end(), previous.begin());
      const auto ended = Clock::now();
      if (hop >= warmup) {
        runTimes[static_cast<size_t>(hop - warmup)] = milliseconds(runEnd - runBegan);
        loopTimes[static_cast<size_t>(hop - warmup)] = milliseconds(ended - began);
      }
    }
    require(maximumClosure < 2e-6, "Physical mixture closure failed");
    std::cout.precision(12);
    std::cout << "{\"status\":\"pass\",\"onnxruntime_version\":\""
              << OrtGetApiBase()->GetVersionString()
              << "\",\"intra_op_threads\":1,\"inter_op_threads\":1,"
                 "\"provider\":\"CPU\",\"execution\":\"sequential\","
                 "\"spinning\":false,\"preallocated_tensors\":true,"
                 "\"warmup_hops\":" << warmup << ",\"measured_hops\":" << measured
              << ",\"maximum_closure\":" << maximumClosure << ",\"run\":";
    statistics(std::move(runTimes));
    std::cout << ",\"run_with_copy_and_state_checks\":";
    statistics(std::move(loopTimes));
    std::cout << ",\"thread_priority\":\"" << priority
              << "\",\"plugin_queue_measured\":false,\"native_host_qualified\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
