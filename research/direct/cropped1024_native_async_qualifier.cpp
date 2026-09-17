// Source preparation for the listened cropped1024 raw-four L1 +250 graph.
//
// Scope: Linux/WSL CPU ONNX Runtime 1.26.0, a paced 256-sample callback,
// and exactly one asynchronous worker/queue hop. Graph256 + queue256 =512.
// This is not plugin, DAW, bus, packaging, Windows, or macOS qualification.
// Compilation, graph parity and actual bounded timing remain separate steps.
// Required future identities: --model-sha256, --model-bytes, --ort-sha256.
// Reuses the hardened atomic queue from research/c91_native_async_qualifier.cpp;
// the legacy CV protocol is omitted. Complete graph outputs are never remixed.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

constexpr int kSampleRate = 44100;
constexpr int kHopSamples = 256;
constexpr int kChannels = 2;
constexpr int kStems = 4;
constexpr int kHistorySamples = 768;
constexpr int kHiddenLayers = 2;
constexpr int kHiddenSize = 1000;
constexpr size_t kQueueSlots = 16;
constexpr size_t kAudioElements = kChannels * kHopSamples;
constexpr size_t kSeparatedElements = kStems * kChannels * kHopSamples;
constexpr size_t kHistoryElements = kChannels * kHistorySamples;
constexpr size_t kTailElements = kStems * kChannels * kHopSamples;
constexpr size_t kHiddenElements = kHiddenLayers * kHiddenSize;
constexpr double kDeadlineMilliseconds =
    1000.0 * static_cast<double>(kHopSamples) /
    static_cast<double>(kSampleRate);
constexpr std::string_view kFamily = "ola-cropped1024-hann512-hop256-v1";
// These identify the already completed and listened checkpoint, not a future graph.
constexpr std::string_view kCheckpointSha =
    "ac46729e5e4d379b09914a6e40ae927e09089b43fd4eef219ae7e034f355da65";
constexpr std::string_view kModelStateSha =
    "a12c215810026c603a1fd394383c1646219b8b3f764ebe9c2a83856404443aa4";

using AudioChunk = std::array<float, kAudioElements>;
using SeparatedChunk = std::array<float, kSeparatedElements>;
using AlignedChunk = std::array<float, kAudioElements>;
constexpr std::array<const char*, 4> kStemNames = {"drums", "bass", "vocals", "other"};
constexpr double kParityMaximumAbsoluteError = 1.0e-4;
constexpr double kParityMaximumCallbackRmsError = 1.0e-5;
static_assert(sizeof(float) == 4U && std::numeric_limits<float>::is_iec559,
              "WAV and ONNX parity require IEEE binary32");

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::string jsonEscape(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '\"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

uint32_t rotateRight(uint32_t value, unsigned int count) {
  return (value >> count) | (value << (32U - count));
}

class Sha256 final {
 public:
  Sha256() { reset(); }

  void update(const uint8_t* data, size_t size) {
    for (size_t index = 0; index < size; ++index) {
      block_[blockSize_++] = data[index];
      if (blockSize_ == block_.size()) {
        transform();
        bitCount_ += 512U;
        blockSize_ = 0;
      }
    }
  }

  std::string finish() {
    const uint64_t totalBits =
        bitCount_ + static_cast<uint64_t>(blockSize_) * 8U;
    block_[blockSize_++] = 0x80U;
    if (blockSize_ > 56U) {
      while (blockSize_ < 64U) block_[blockSize_++] = 0U;
      transform();
      blockSize_ = 0;
    }
    while (blockSize_ < 56U) block_[blockSize_++] = 0U;
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[blockSize_++] = static_cast<uint8_t>(totalBits >> shift);
    }
    transform();

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const uint32_t word : state_) output << std::setw(8) << word;
    return output.str();
  }

 private:
  void reset() {
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    block_.fill(0U);
    blockSize_ = 0;
    bitCount_ = 0;
  }

  void transform() {
    static constexpr std::array<uint32_t, 64> constants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<uint32_t, 64> schedule{};
    for (size_t index = 0; index < 16U; ++index) {
      const size_t offset = index * 4U;
      schedule[index] = (static_cast<uint32_t>(block_[offset]) << 24U) |
                        (static_cast<uint32_t>(block_[offset + 1U]) << 16U) |
                        (static_cast<uint32_t>(block_[offset + 2U]) << 8U) |
                        static_cast<uint32_t>(block_[offset + 3U]);
    }
    for (size_t index = 16U; index < schedule.size(); ++index) {
      const uint32_t s0 = rotateRight(schedule[index - 15U], 7U) ^
                          rotateRight(schedule[index - 15U], 18U) ^
                          (schedule[index - 15U] >> 3U);
      const uint32_t s1 = rotateRight(schedule[index - 2U], 17U) ^
                          rotateRight(schedule[index - 2U], 19U) ^
                          (schedule[index - 2U] >> 10U);
      schedule[index] = schedule[index - 16U] + s0 +
                        schedule[index - 7U] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (size_t index = 0; index < schedule.size(); ++index) {
      const uint32_t sum1 = rotateRight(e, 6U) ^ rotateRight(e, 11U) ^
                            rotateRight(e, 25U);
      const uint32_t choice = (e & f) ^ ((~e) & g);
      const uint32_t temporary1 =
          h + sum1 + choice + constants[index] + schedule[index];
      const uint32_t sum0 = rotateRight(a, 2U) ^ rotateRight(a, 13U) ^
                            rotateRight(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_{};
  std::array<uint8_t, 64> block_{};
  size_t blockSize_{0};
  uint64_t bitCount_{0};
};

std::string sha256File(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot hash file: " + path.string());
  Sha256 sha;
  std::vector<char> buffer(1024U * 1024U);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      sha.update(reinterpret_cast<const uint8_t*>(buffer.data()),
                 static_cast<size_t>(count));
    }
  }
  require(input.eof(), "failed while hashing file: " + path.string());
  return sha.finish();
}

std::filesystem::path loadedOrtPath() {
  std::ifstream maps("/proc/self/maps");
  require(maps.good(), "cannot inspect /proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    const size_t marker = line.find("libonnxruntime.so");
    if (marker == std::string::npos) continue;
    const size_t pathStart = line.find('/');
    if (pathStart != std::string::npos) {
      return std::filesystem::canonical(line.substr(pathStart));
    }
  }
  fail("loaded libonnxruntime path was not found in /proc/self/maps");
}

std::string metadataValue(const Ort::Session& session, const char* key) {
  Ort::AllocatorWithDefaultOptions allocator;
  const Ort::ModelMetadata metadata = session.GetModelMetadata();
  auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
  return value ? std::string(value.get()) : std::string();
}

void validateTensor(const Ort::Session& session, bool input, size_t index,
                    std::string_view expectedName,
                    const std::vector<int64_t>& expectedShape) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = input ? session.GetInputNameAllocated(index, allocator)
                    : session.GetOutputNameAllocated(index, allocator);
  require(name && std::string_view(name.get()) == expectedName,
          "ONNX tensor name/order changed at index " +
              std::to_string(index));
  const auto type = input ? session.GetInputTypeInfo(index)
                          : session.GetOutputTypeInfo(index);
  const auto tensor = type.GetTensorTypeAndShapeInfo();
  require(tensor.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
          "ONNX tensor is not float32: " + std::string(expectedName));
  require(tensor.GetShape() == expectedShape,
          "ONNX tensor shape changed: " + std::string(expectedName));
}

void validateModelContract(const Ort::Session& session) {
  require(session.GetInputCount() == 5U, "cropped ONNX input count changed");
  require(session.GetOutputCount() == 5U, "cropped ONNX output count changed");
  validateTensor(session, true, 0, "audio_chunk", {1, 2, 256});
  validateTensor(session, true, 1, "audio_history", {1, 2, 768});
  validateTensor(session, true, 2, "fusion_hidden", {2, 1, 1000});
  validateTensor(session, true, 3, "spectral_numerator_tail", {1, 4, 2, 256});
  validateTensor(session, true, 4, "waveform_tail", {1, 4, 2, 256});
  validateTensor(session, false, 0, "separated_chunk", {1, 4, 2, 256});
  validateTensor(session, false, 1, "next_audio_history", {1, 2, 768});
  validateTensor(session, false, 2, "next_fusion_hidden", {2, 1, 1000});
  validateTensor(session, false, 3, "next_spectral_numerator_tail", {1, 4, 2, 256});
  validateTensor(session, false, 4, "next_waveform_tail", {1, 4, 2, 256});
  const auto expect = [&](const char* key, std::string_view value) {
    require(metadataValue(session, key) == value,
            "cropped ONNX metadata changed: " + std::string(key));
  };
  expect("hs_tasnet.kind", "cropped1024_ola");
  expect("hs_tasnet.mode", "streaming");
  expect("hs_tasnet.architecture_version", kFamily);
  expect("hs_tasnet.state_family", kFamily);
  expect("hs_tasnet.state_interchangeable_with_old_ola512", "false");
  expect("hs_tasnet.sample_rate", "44100");
  expect("hs_tasnet.hop_samples", "256");
  expect("hs_tasnet.analysis_fft_samples", "1024");
  expect("hs_tasnet.carrier_fft_samples", "1024");
  expect("hs_tasnet.synthesis_fft_samples", "1024");
  expect("hs_tasnet.spectral_mask_bins", "513");
  expect("hs_tasnet.spectral_output_crop", "[512,1024]");
  expect("hs_tasnet.synthesis_frame_samples", "512");
  expect("hs_tasnet.waveform_decoder_samples", "512");
  expect("hs_tasnet.analysis_history_samples", "768");
  expect("hs_tasnet.graph_output_delay_samples", "256");
  expect("hs_tasnet.alignment_samples", "256");
  expect("hs_tasnet.future_callbacks_beyond_received_input", "0");
  expect("hs_tasnet.initial_state", "all_zeros");
  expect("hs_tasnet.preroll", "discard_first_output_hop_after_reset");
  expect("hs_tasnet.flush_required", "true");
  expect("hs_tasnet.flush_hops", "1");
  expect("hs_tasnet.external_host_queue_implemented", "false");
  expect("hs_tasnet.intended_external_host_queue_samples", "256");
  expect("hs_tasnet.intended_total_latency_samples", "512");
  expect("hs_tasnet.source_order", "drums,bass,vocals,other");
  expect("hs_tasnet.output_policy",
         "complete deployed four stems; Other = previous physical mixture - sum(unchanged DBV), once");
  expect("hs_tasnet.output_source_scales",
         "[0.5, 0.5, 0.44999998807907104, 0.5600000023841858]");
  expect("hs_tasnet.public_fusion_state_scale", "3.814697265625e-06");
  expect("hs_tasnet.state_names",
         "[\"audio_history\", \"fusion_hidden\", \"spectral_numerator_tail\", \"waveform_tail\"]");
  expect("hs_tasnet.state_shapes",
         "[[1, 2, 768], [2, 1, 1000], [1, 4, 2, 256], [1, 4, 2, 256]]");
  expect("hs_tasnet.checkpoint_sha256", kCheckpointSha);
  expect("hs_tasnet.model_state_sha256", kModelStateSha);
  expect("hs_tasnet.snapshot_step", "250");
  expect("hs_tasnet.training_updates", "2250");
  expect("hs_tasnet.external_data", "false");
}

struct HopResult {
  bool valid{};
  bool allIncomingStatesNonzero{};
  double maximumMixtureError{};
  double nextHistoryMaximumError{};
  bool nextHistoryBitExact{};
};

class StreamingModel final {
 public:
  StreamingModel(const std::filesystem::path& modelPath, int threads)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "cropped1024_native_async"),
        memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                             OrtMemTypeDefault)) {
    Ort::SessionOptions options;
    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(threads);
    options.SetInterOpNumThreads(1);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    session_ = std::make_unique<Ort::Session>(environment_, modelPath.c_str(),
                                            options);
    validateModelContract(*session_);
    initializeValues();
    reset();
  }

  void reset() {
    audio_.fill(0.0F);
    history_.fill(0.0F);
    hidden_.fill(0.0F);
    spectralTail_.fill(0.0F);
    waveformTail_.fill(0.0F);
    separated_.fill(0.0F);
    nextHistory_.fill(0.0F);
    nextHidden_.fill(0.0F);
    nextSpectralTail_.fill(0.0F);
    nextWaveformTail_.fill(0.0F);
    hasPreviousInput_ = false;
  }

  HopResult run(const AudioChunk& input, SeparatedChunk& published,
                AlignedChunk& aligned) {
    audio_ = input;
    const auto nonzero = [](const auto& values) {
      return std::any_of(values.begin(), values.end(),
                         [](float value) { return value != 0.0F; });
    };
    const bool nonzeroIncoming = nonzero(history_) && nonzero(hidden_) &&
                                  nonzero(spectralTail_) && nonzero(waveformTail_);
    // Capture physical alignment before ONNX or the state update. The graph's
    // output corresponds to the last256 samples of the incoming 768 history.
    aligned.fill(0.0F);
    if (hasPreviousInput_) {
      for (size_t channel = 0; channel < kChannels; ++channel) {
        std::copy_n(history_.begin() + channel * kHistorySamples +
                        (kHistorySamples - kHopSamples),
                    kHopSamples, aligned.begin() + channel * kHopSamples);
      }
    }
    std::array<float, kHistoryElements> expectedHistory{};
    for (size_t channel = 0; channel < kChannels; ++channel) {
      std::copy_n(history_.begin() + channel * kHistorySamples + kHopSamples,
                  kHistorySamples - kHopSamples,
                  expectedHistory.begin() + channel * kHistorySamples);
      std::copy_n(audio_.begin() + channel * kHopSamples, kHopSamples,
                  expectedHistory.begin() + channel * kHistorySamples +
                      (kHistorySamples - kHopSamples));
    }
    session_->Run(runOptions_, inputNames_.data(), inputValues_.data(),
                  inputValues_.size(), outputNames_.data(),
                  outputValues_.data(), outputValues_.size());
    const auto finite = [](const auto& values) {
      return std::all_of(values.begin(), values.end(),
                         [](float value) { return std::isfinite(value); });
    };
    require(finite(separated_) && finite(nextHistory_) && finite(nextHidden_) &&
                finite(nextSpectralTail_) && finite(nextWaveformTail_),
            "cropped ONNX returned a non-finite output or state");
    HopResult result;
    result.valid = hasPreviousInput_;
    result.allIncomingStatesNonzero = nonzeroIncoming;
    if (result.valid) {
      for (size_t channel = 0; channel < kChannels; ++channel) {
        for (size_t sample = 0; sample < kHopSamples; ++sample) {
          float sum = 0.0F;
          for (size_t stem = 0; stem < kStems; ++stem) {
            sum += separated_[(stem * kChannels + channel) * kHopSamples +
                              sample];
          }
          result.maximumMixtureError = std::max(
              result.maximumMixtureError,
              std::abs(static_cast<double>(sum) -
                       static_cast<double>(aligned[channel * kHopSamples +
                                                   sample])));
        }
      }
      // Export already constructs deployed Other exactly once. Preserve all
      // four graph outputs; a host residual rewrite would hide graph errors.
      published = separated_;
    } else {
      // First graph output after every reset is invalid, even if biases emit
      // nonzero samples. The external queue adds one earlier invalid hop.
      published.fill(0.0F);
    }
    require(result.maximumMixtureError <= 1.0e-6,
            "deployed graph mixture reconstruction exceeded 1e-6");
    result.nextHistoryBitExact =
        std::memcmp(nextHistory_.data(), expectedHistory.data(),
                    expectedHistory.size() * sizeof(float)) == 0;
    for (size_t index = 0; index < expectedHistory.size(); ++index) {
      result.nextHistoryMaximumError = std::max(
          result.nextHistoryMaximumError,
          std::abs(static_cast<double>(nextHistory_[index]) -
                   static_cast<double>(expectedHistory[index])));
    }
    history_ = nextHistory_;
    hidden_ = nextHidden_;
    spectralTail_ = nextSpectralTail_;
    waveformTail_ = nextWaveformTail_;
    hasPreviousInput_ = true;
    return result;
  }

 private:
  void initializeValues() {
    static constexpr std::array<int64_t, 3> audioShape = {1, 2, 256};
    static constexpr std::array<int64_t, 3> historyShape = {1, 2, 768};
    static constexpr std::array<int64_t, 3> hiddenShape = {2, 1, 1000};
    static constexpr std::array<int64_t, 4> tailShape = {1, 4, 2, 256};
    static constexpr std::array<int64_t, 4> separatedShape = {1, 4, 2, 256};
    inputValues_.reserve(5U);
    outputValues_.reserve(5U);
    const auto tensor = [&](auto& data, const auto& shape) {
      return Ort::Value::CreateTensor<float>(memoryInfo_, data.data(),
          data.size(), shape.data(), shape.size());
    };
    inputValues_.emplace_back(tensor(audio_, audioShape));
    inputValues_.emplace_back(tensor(history_, historyShape));
    inputValues_.emplace_back(tensor(hidden_, hiddenShape));
    inputValues_.emplace_back(tensor(spectralTail_, tailShape));
    inputValues_.emplace_back(tensor(waveformTail_, tailShape));
    outputValues_.emplace_back(tensor(separated_, separatedShape));
    outputValues_.emplace_back(tensor(nextHistory_, historyShape));
    outputValues_.emplace_back(tensor(nextHidden_, hiddenShape));
    outputValues_.emplace_back(tensor(nextSpectralTail_, tailShape));
    outputValues_.emplace_back(tensor(nextWaveformTail_, tailShape));
  }

  Ort::Env environment_;
  Ort::MemoryInfo memoryInfo_;
  std::unique_ptr<Ort::Session> session_;
  AudioChunk audio_{};
  std::array<float, kHistoryElements> history_{};
  std::array<float, kHiddenElements> hidden_{};
  std::array<float, kTailElements> spectralTail_{};
  std::array<float, kTailElements> waveformTail_{};
  SeparatedChunk separated_{};
  std::array<float, kHistoryElements> nextHistory_{};
  std::array<float, kHiddenElements> nextHidden_{};
  std::array<float, kTailElements> nextSpectralTail_{};
  std::array<float, kTailElements> nextWaveformTail_{};
  bool hasPreviousInput_{};
  std::array<const char*, 5> inputNames_ = {
      "audio_chunk", "audio_history", "fusion_hidden",
      "spectral_numerator_tail", "waveform_tail"};
  std::array<const char*, 5> outputNames_ = {
      "separated_chunk", "next_audio_history", "next_fusion_hidden",
      "next_spectral_numerator_tail", "next_waveform_tail"};
  std::vector<Ort::Value> inputValues_;
  std::vector<Ort::Value> outputValues_;
  Ort::RunOptions runOptions_{nullptr};
};

constexpr std::string_view queueModeName() {
  return "hardened_atomic_slot_poll_sleep_100us";
}

struct Record {
  TimePoint scheduledCallback{};
  TimePoint deadline{};
  TimePoint inputPublished{};
  TimePoint runBegin{};
  TimePoint runEnd{};
  TimePoint prePublicationLowerBound{};
  std::atomic<int64_t> postPublicationNanoseconds{0};
  std::atomic<int64_t> consumerBoundaryObservationNanoseconds{0};
  bool submitted{};
  bool inferenceSucceeded{};
  std::string inferenceError;
  bool staleDiscarded{};
  bool outputValid{};
  bool allIncomingStatesNonzero{};
  double maximumMixtureError{};
  double nextHistoryMaximumError{};
  bool nextHistoryBitExact{};
  bool callbackChecked{};
  bool availableAtActualBoundary{};
  bool consumerSequencePass{};
  bool consumerValidityPass{};
  bool consumerAlignmentPass{};
  bool consumerPdcPass{};
  bool consumerReconstructionPass{};
  double consumerMaximumMixtureError{};
  std::atomic<bool> workerFinished{false};
};

enum class SlotState : uint8_t {
  Empty,
  Writing,
  Ready,
  Processing,
  Processed,
  Reading,
};

struct Slot {
  static constexpr uint64_t makeControl(uint32_t epoch, SlotState state) {
    return (static_cast<uint64_t>(epoch) << 32U) |
           static_cast<uint64_t>(state);
  }
  static constexpr SlotState stateFromControl(uint64_t control) {
    return static_cast<SlotState>(control & 0xffU);
  }
  static constexpr uint32_t epochFromControl(uint64_t control) {
    return static_cast<uint32_t>(control >> 32U);
  }

  AudioChunk input{};
  SeparatedChunk output{};
  AlignedChunk aligned{};
  uint64_t sequence{};
  Record* record{};
  std::atomic<uint64_t> control{makeControl(0U, SlotState::Empty)};
};

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "packed queue ownership must be lock-free");

struct WorkerPlatform {
  bool affinityAttempted{};
  int affinityError{};
  std::string affinityObserved;
  bool priorityAttempted{};
  int priorityError{};
  int schedulingPolicy{};
  int schedulingPriority{};
};

std::string cpuSetString(const cpu_set_t& set) {
  std::ostringstream output;
  bool first = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &set)) continue;
    if (!first) output << ',';
    output << cpu;
    first = false;
  }
  return output.str();
}

struct QueueTelemetry {
  std::atomic<uint64_t> queueFullDrops{0};
  std::atomic<uint64_t> staleDiscards{0};
  std::atomic<uint64_t> inFlightResetDiscards{0};
  std::atomic<uint64_t> lateTimelineDiscards{0};
  std::atomic<uint64_t> orderingFailures{0};
  std::atomic<uint64_t> sequenceGaps{0};
  std::atomic<uint64_t> resetRequests{0};
  std::atomic<uint64_t> resetAcknowledgements{0};
  std::atomic<uint64_t> consumerClaims{0};
  std::atomic<uint64_t> consumerReleases{0};
};

struct OutputClaim {
  Slot* slot{};
  size_t index{};
};

struct ResetEvent {
  uint32_t epoch{};
  double callMilliseconds{};
  bool releasedReading{};
  bool releasedWriting{};
};

class AsyncQueue final {
 public:
  AsyncQueue(const std::filesystem::path& modelPath,
             int threads, std::vector<int> cpus)
      : modelPath_(modelPath), threads_(threads),
        requestedCpus_(std::move(cpus)) {}

  ~AsyncQueue() { stop(); }

  AsyncQueue(const AsyncQueue&) = delete;
  AsyncQueue& operator=(const AsyncQueue&) = delete;

  void start() {
    worker_ = std::thread(&AsyncQueue::workerMain, this);
    std::unique_lock<std::mutex> lock(startMutex_);
    startCv_.wait(lock, [this] {
      return started_.load(std::memory_order_acquire) ||
             workerFailed_.load(std::memory_order_acquire);
    });
    rethrowWorkerFailure();
  }

  void stop() {
    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) worker_.join();
  }

  bool submit(const AudioChunk& input, uint64_t sequence, uint32_t epoch,
              Record& record) {
    const size_t index = writeIndex_.load(std::memory_order_acquire);
    Slot& slot = slots_[index];

    uint64_t observed = slot.control.load(std::memory_order_acquire);
    while (true) {
      const SlotState state = Slot::stateFromControl(observed);
      const uint32_t slotEpoch = Slot::epochFromControl(observed);
      const bool claimable =
          state == SlotState::Empty ||
          ((state == SlotState::Ready || state == SlotState::Processed) &&
           slotEpoch != epoch);
      if (!claimable) {
        telemetry_.queueFullDrops.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      const uint64_t writing = Slot::makeControl(epoch, SlotState::Writing);
      if (slot.control.compare_exchange_weak(
              observed, writing, std::memory_order_acquire,
              std::memory_order_acquire)) {
        break;
      }
    }
    slot.input = input;
    slot.sequence = sequence;
    slot.record = &record;
    record.inputPublished = Clock::now();
    record.submitted = true;
    uint64_t expected = Slot::makeControl(epoch, SlotState::Writing);
    if (epoch != currentEpoch()) {
      static_cast<void>(slot.control.compare_exchange_strong(
          expected, Slot::makeControl(epoch, SlotState::Empty),
          std::memory_order_release, std::memory_order_relaxed));
      record.submitted = false;
      return false;
    }
    if (!slot.control.compare_exchange_strong(
            expected, Slot::makeControl(epoch, SlotState::Ready),
            std::memory_order_release, std::memory_order_relaxed)) {
      record.submitted = false;
      return false;
    }
    writeIndex_.store((index + 1U) % kQueueSlots, std::memory_order_release);
    return true;
  }

  std::optional<OutputClaim> claimExpected(uint32_t epoch,
                                           uint64_t dueSequence) {
    if (epoch != currentEpoch()) return std::nullopt;
    while (true) {
      const size_t index = consumeIndex_.load(std::memory_order_acquire);
      Slot& slot = slots_[index];

      uint64_t observed = slot.control.load(std::memory_order_acquire);
      if (Slot::stateFromControl(observed) != SlotState::Processed) {
        return std::nullopt;
      }
      const uint32_t slotEpoch = Slot::epochFromControl(observed);
      if (slotEpoch != epoch) {
        const uint64_t empty = Slot::makeControl(slotEpoch, SlotState::Empty);
        if (!slot.control.compare_exchange_strong(
                observed, empty, std::memory_order_release,
                std::memory_order_acquire)) {
          continue;
        }
        consumeIndex_.store((index + 1U) % kQueueSlots,
                            std::memory_order_release);
        telemetry_.staleDiscards.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      if (slot.sequence > dueSequence) {
        telemetry_.orderingFailures.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
      }
      const uint64_t reading = Slot::makeControl(slotEpoch, SlotState::Reading);
      if (!slot.control.compare_exchange_strong(
              observed, reading, std::memory_order_acquire,
              std::memory_order_acquire)) {
        continue;
      }
      if (slot.sequence < dueSequence) {
        releaseClaim({&slot, index});
        telemetry_.lateTimelineDiscards.fetch_add(1,
                                                  std::memory_order_relaxed);
        continue;
      }
      telemetry_.consumerClaims.fetch_add(1, std::memory_order_relaxed);
      return OutputClaim{&slot, index};
    }
  }

  bool releaseClaim(const OutputClaim& claim) {
    require(claim.slot != nullptr, "cannot release a null output claim");
    bool released = false;
    uint64_t observed = claim.slot->control.load(std::memory_order_acquire);
    if (Slot::stateFromControl(observed) == SlotState::Reading) {
      const uint64_t empty = Slot::makeControl(
          Slot::epochFromControl(observed), SlotState::Empty);
      released = claim.slot->control.compare_exchange_strong(
          observed, empty, std::memory_order_release,
          std::memory_order_acquire);
    }
    if (released) {
      consumeIndex_.store((claim.index + 1U) % kQueueSlots,
                          std::memory_order_release);
      telemetry_.consumerReleases.fetch_add(1, std::memory_order_relaxed);
    }
    return released;
  }

  ResetEvent resetNonBlocking() {
    const TimePoint begin = Clock::now();
    ResetEvent event;
    const size_t consumeIndex =
        consumeIndex_.load(std::memory_order_acquire);
    Slot& consumed = slots_[consumeIndex];
    uint64_t observed = consumed.control.load(std::memory_order_acquire);
    if (Slot::stateFromControl(observed) == SlotState::Reading) {
      const uint64_t empty = Slot::makeControl(
          Slot::epochFromControl(observed), SlotState::Empty);
      event.releasedReading = consumed.control.compare_exchange_strong(
          observed, empty, std::memory_order_release,
          std::memory_order_relaxed);
    }

    const size_t startIndex = writeIndex_.load(std::memory_order_acquire);
    Slot& writing = slots_[startIndex];
    observed = writing.control.load(std::memory_order_acquire);
    if (Slot::stateFromControl(observed) == SlotState::Writing) {
      const uint64_t empty = Slot::makeControl(
          Slot::epochFromControl(observed), SlotState::Empty);
      event.releasedWriting = writing.control.compare_exchange_strong(
          observed, empty, std::memory_order_release,
          std::memory_order_relaxed);
    }

    consumeIndex_.store(startIndex, std::memory_order_release);
    uint64_t epochControl =
        atomicEpochControl_.load(std::memory_order_acquire);
    while (true) {
      event.epoch = epochFromEpochControl(epochControl) + 1U;
      const uint64_t desired = makeEpochControl(event.epoch, startIndex);
      if (atomicEpochControl_.compare_exchange_weak(
              epochControl, desired, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        break;
      }
    }
    telemetry_.resetRequests.fetch_add(1, std::memory_order_relaxed);
    event.callMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return event;
  }

  void waitForResetAcknowledgement(uint32_t epoch) const {
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (workerAcknowledgedEpoch_.load(std::memory_order_acquire) < epoch) {
      rethrowWorkerFailure();
      require(Clock::now() < deadline,
              "worker reset acknowledgement timed out");
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  uint32_t currentEpoch() const {
    return epochFromEpochControl(
        atomicEpochControl_.load(std::memory_order_acquire));
  }
  uint32_t workerAcknowledgedEpoch() const {
    return workerAcknowledgedEpoch_.load(std::memory_order_acquire);
  }
  uint64_t workerInFlightSequence() const {
    return workerInFlightSequence_.load(std::memory_order_acquire);
  }
  const QueueTelemetry& telemetry() const { return telemetry_; }
  const WorkerPlatform& platform() const { return platform_; }

  void rethrowWorkerFailure() const {
    if (workerFailed_.load(std::memory_order_acquire))
      std::rethrow_exception(workerFailure_);
  }

 private:
  static constexpr uint64_t makeEpochControl(uint32_t epoch,
                                             size_t startIndex) {
    return (static_cast<uint64_t>(epoch) << 32U) |
           static_cast<uint64_t>(static_cast<uint32_t>(startIndex));
  }
  static constexpr uint32_t epochFromEpochControl(uint64_t control) {
    return static_cast<uint32_t>(control >> 32U);
  }
  static constexpr size_t startIndexFromEpochControl(uint64_t control) {
    return static_cast<size_t>(static_cast<uint32_t>(control));
  }

  bool slotReady(size_t index) const {
    const Slot& slot = slots_[index];
    return Slot::stateFromControl(
        slot.control.load(std::memory_order_acquire)) == SlotState::Ready;
  }

  void configureWorkerPlatform() {
    if (!requestedCpus_.empty()) {
      platform_.affinityAttempted = true;
      cpu_set_t requested;
      CPU_ZERO(&requested);
      for (const int cpu : requestedCpus_) {
        require(cpu >= 0 && cpu < CPU_SETSIZE,
                "requested CPU index is outside cpu_set_t");
        CPU_SET(cpu, &requested);
      }
      platform_.affinityError =
          pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested);
    }
    cpu_set_t observed;
    CPU_ZERO(&observed);
    const int affinityRead =
        pthread_getaffinity_np(pthread_self(), sizeof(observed), &observed);
    require(affinityRead == 0, "pthread_getaffinity_np failed");
    platform_.affinityObserved = cpuSetString(observed);

    // Scheduling is observed, never elevated. A measured deadline result is
    // separate from an RT scheduler guarantee or target-host qualification.
    platform_.priorityAttempted = false;
    platform_.priorityError = 0;
    sched_param actual{};
    require(pthread_getschedparam(pthread_self(), &platform_.schedulingPolicy,
                                  &actual) == 0,
            "pthread_getschedparam failed");
    platform_.schedulingPriority = actual.sched_priority;
  }

  bool reclaimAtomicStale(uint64_t epochControl) {
    const uint32_t epoch = epochFromEpochControl(epochControl);
    for (Slot& slot : slots_) {
      uint64_t observed = slot.control.load(std::memory_order_acquire);
      const SlotState state = Slot::stateFromControl(observed);
      const uint32_t slotEpoch = Slot::epochFromControl(observed);
      if (slotEpoch == epoch ||
          (state != SlotState::Ready && state != SlotState::Processed)) {
        continue;
      }
      if (atomicEpochControl_.load(std::memory_order_acquire) != epochControl) {
        return false;
      }
      Record* const record = slot.record;
      const uint64_t empty = Slot::makeControl(slotEpoch, SlotState::Empty);
      if (slot.control.compare_exchange_strong(
              observed, empty, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        if (record != nullptr &&
            !record->workerFinished.load(std::memory_order_acquire)) {
          record->staleDiscarded = true;
          record->workerFinished.store(true, std::memory_order_release);
        }
        telemetry_.staleDiscards.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return atomicEpochControl_.load(std::memory_order_acquire) == epochControl;
  }

  void finishStaleRecord(Record& record, bool inFlight) {
    record.staleDiscarded = true;
    record.workerFinished.store(true, std::memory_order_release);
    telemetry_.staleDiscards.fetch_add(1, std::memory_order_relaxed);
    if (inFlight) {
      telemetry_.inFlightResetDiscards.fetch_add(1,
                                                 std::memory_order_relaxed);
    }
  }

  void installHopResult(Record& record, const HopResult& hop) {
    record.inferenceSucceeded = true;
    record.outputValid = hop.valid;
    record.allIncomingStatesNonzero = hop.allIncomingStatesNonzero;
    record.maximumMixtureError = hop.maximumMixtureError;
    record.nextHistoryMaximumError = hop.nextHistoryMaximumError;
    record.nextHistoryBitExact = hop.nextHistoryBitExact;
  }

  void processAtomicSlot(StreamingModel& model, size_t index,
                         uint64_t epochControl,
                         bool& havePreviousSequence,
                         uint64_t& previousSequence) {
    Slot& slot = slots_[index];
    uint64_t requestControl = slot.control.load(std::memory_order_acquire);
    if (Slot::stateFromControl(requestControl) != SlotState::Ready) return;
    const uint32_t requestEpoch = Slot::epochFromControl(requestControl);
    const uint64_t processing =
        Slot::makeControl(requestEpoch, SlotState::Processing);
    if (!slot.control.compare_exchange_strong(
            requestControl, processing, std::memory_order_acquire,
            std::memory_order_acquire)) {
      return;
    }
    Record& record = *slot.record;
    if (requestEpoch != epochFromEpochControl(epochControl) ||
        atomicEpochControl_.load(std::memory_order_acquire) != epochControl) {
      uint64_t expected = processing;
      static_cast<void>(slot.control.compare_exchange_strong(
          expected, Slot::makeControl(requestEpoch, SlotState::Empty),
          std::memory_order_release, std::memory_order_relaxed));
      finishStaleRecord(record, false);
      return;
    }
    if (havePreviousSequence && slot.sequence != previousSequence + 1U) {
      telemetry_.sequenceGaps.fetch_add(1, std::memory_order_relaxed);
      model.reset();
      havePreviousSequence = false;
    }
    if (atomicEpochControl_.load(std::memory_order_acquire) != epochControl) {
      uint64_t expected = processing;
      static_cast<void>(slot.control.compare_exchange_strong(
          expected, Slot::makeControl(requestEpoch, SlotState::Empty),
          std::memory_order_release, std::memory_order_relaxed));
      finishStaleRecord(record, false);
      return;
    }
    workerInFlightSequence_.store(slot.sequence, std::memory_order_release);
    record.runBegin = Clock::now();
    HopResult hop;
    bool inferenceOk = false;
    try {
      hop = model.run(slot.input, slot.output, slot.aligned);
      inferenceOk = true;
    } catch (...) {
      try {
        throw;
      } catch (const std::exception& error) {
        record.inferenceError = error.what();
      } catch (...) {
        record.inferenceError = "nonstandard inference exception";
      }
      slot.output.fill(0.0F);
      slot.aligned.fill(0.0F);
      model.reset();
      havePreviousSequence = false;
    }
    record.runEnd = Clock::now();
    workerInFlightSequence_.store(kNoInFlightSequence,
                                  std::memory_order_release);
    if (atomicEpochControl_.load(std::memory_order_acquire) != epochControl) {
      uint64_t expected = processing;
      static_cast<void>(slot.control.compare_exchange_strong(
          expected, Slot::makeControl(requestEpoch, SlotState::Empty),
          std::memory_order_release, std::memory_order_relaxed));
      finishStaleRecord(record, true);
      return;
    }
    if (inferenceOk) {
      installHopResult(record, hop);
      previousSequence = slot.sequence;
      havePreviousSequence = true;
    }
    record.prePublicationLowerBound = Clock::now();
    uint64_t expected = processing;
    if (slot.control.compare_exchange_strong(
            expected, Slot::makeControl(requestEpoch, SlotState::Processed),
            std::memory_order_release, std::memory_order_relaxed)) {
      record.postPublicationNanoseconds.store(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              Clock::now().time_since_epoch())
              .count(),
          std::memory_order_release);
      record.workerFinished.store(true, std::memory_order_release);
      readIndex_.store((index + 1U) % kQueueSlots,
                       std::memory_order_release);
    } else {
      finishStaleRecord(record, true);
    }
  }

  void workerMain() noexcept {
    try {
      configureWorkerPlatform();
      StreamingModel model(modelPath_, threads_);
      {
        std::lock_guard<std::mutex> lock(startMutex_);
        started_.store(true, std::memory_order_release);
      }
      startCv_.notify_all();
      bool havePreviousSequence = false;
      uint64_t previousSequence = 0;
      uint64_t lastAtomicEpochControl =
          atomicEpochControl_.load(std::memory_order_acquire);
      workerAcknowledgedEpoch_.store(currentEpoch(),
                                     std::memory_order_release);

      const auto synchronizeAtomicEpoch = [&]() {
        while (true) {
          const uint64_t current =
              atomicEpochControl_.load(std::memory_order_acquire);
          if (current == lastAtomicEpochControl) return current;
          model.reset();
          havePreviousSequence = false;
          readIndex_.store(startIndexFromEpochControl(current),
                           std::memory_order_release);
          lastAtomicEpochControl = current;
          if (reclaimAtomicStale(current)) {
            const uint32_t epoch = epochFromEpochControl(current);
            workerAcknowledgedEpoch_.store(epoch, std::memory_order_release);
            telemetry_.resetAcknowledgements.fetch_add(
                1, std::memory_order_relaxed);
            return current;
          }
        }
      };

      while (!stop_.load(std::memory_order_acquire)) {
        uint64_t atomicSnapshot = lastAtomicEpochControl;
        atomicSnapshot = synchronizeAtomicEpoch();
        const size_t index = readIndex_.load(std::memory_order_acquire);
        if (!slotReady(index)) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          continue;
        }
        atomicSnapshot = synchronizeAtomicEpoch();
        processAtomicSlot(model, index, atomicSnapshot,
                          havePreviousSequence, previousSequence);
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(startMutex_);
        workerFailure_ = std::current_exception();
        workerFailed_.store(true, std::memory_order_release);
      }
      startCv_.notify_all();
    }
  }

  std::filesystem::path modelPath_;
  int threads_;
  std::vector<int> requestedCpus_;
  std::array<Slot, kQueueSlots> slots_{};
  std::atomic<size_t> writeIndex_{0};
  std::atomic<size_t> readIndex_{0};
  std::atomic<size_t> consumeIndex_{0};
  std::atomic<uint64_t> atomicEpochControl_{makeEpochControl(0U, 0U)};
  std::atomic<uint32_t> workerAcknowledgedEpoch_{0};
  static constexpr uint64_t kNoInFlightSequence =
      std::numeric_limits<uint64_t>::max();
  std::atomic<uint64_t> workerInFlightSequence_{kNoInFlightSequence};
  std::atomic<bool> stop_{false};
  std::atomic<bool> started_{false};
  std::thread worker_;
  mutable std::mutex startMutex_;
  std::condition_variable startCv_;
  std::exception_ptr workerFailure_;
  std::atomic<bool> workerFailed_{false};
  QueueTelemetry telemetry_;
  WorkerPlatform platform_;
};

// Qualification/consumer-thread scopes declare this AFTER their records. If a
// timeout or another exception unwinds the scope, join before those records die.
// The worker never constructs this guard, so this cannot join the worker itself.
class StopWorkerOnUnwind final {
 public:
  explicit StopWorkerOnUnwind(AsyncQueue& queue) noexcept
      : queue_(queue), initialExceptions_(std::uncaught_exceptions()) {}
  ~StopWorkerOnUnwind() noexcept {
    if (std::uncaught_exceptions() > initialExceptions_) queue_.stop();
  }
  StopWorkerOnUnwind(const StopWorkerOnUnwind&) = delete;
  StopWorkerOnUnwind& operator=(const StopWorkerOnUnwind&) = delete;

 private:
  AsyncQueue& queue_;
  const int initialExceptions_;
};

AudioChunk makeAudioChunk(int64_t firstSample) {
  AudioChunk chunk{};
  constexpr double twoPi = 6.28318530717958647692;
  for (int sample = 0; sample < kHopSamples; ++sample) {
    const double time = static_cast<double>(firstSample + sample) /
                        static_cast<double>(kSampleRate);
    chunk[static_cast<size_t>(sample)] = static_cast<float>(
        0.17 * std::sin(twoPi * 173.0 * time) +
        0.09 * std::cos(twoPi * 997.0 * time));
    chunk[static_cast<size_t>(kHopSamples + sample)] = static_cast<float>(
        -0.13 * std::cos(twoPi * 251.0 * time) +
        0.07 * std::sin(twoPi * 1301.0 * time));
  }
  return chunk;
}

struct StereoWave {
  uint16_t format{};
  uint16_t bitsPerSample{};
  uint64_t fileFrames{};
  uint64_t framesRead{};
  std::vector<float> interleaved;
};

uint16_t little16(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8U);
}

uint32_t little32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

void readExact(std::ifstream& input, uint8_t* output, size_t bytes) {
  input.read(reinterpret_cast<char*>(output), static_cast<std::streamsize>(bytes));
  require(input.gcount() == static_cast<std::streamsize>(bytes),
          "truncated WAV data");
}

StereoWave readStereoWave(const std::filesystem::path& path,
                          uint64_t requestedFrames = 0U) {
  // Decode only reviewed RIFF/WAVE PCM16 or IEEE FLOAT32 stereo44100.
  // PCM16/32768 is exactly representable in FP32; no normalization or resampling.
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open WAV: " + path.string());
  const uint64_t fileBytes = std::filesystem::file_size(path);
  std::array<uint8_t, 12> header{};
  readExact(input, header.data(), header.size());
  require(std::memcmp(header.data(), "RIFF", 4U) == 0 &&
              std::memcmp(header.data() + 8U, "WAVE", 4U) == 0,
          "only little-endian RIFF/WAVE is supported");
  const uint64_t riffEnd = 8U + little32(header.data() + 4U);
  require(riffEnd >= 12U && riffEnd <= fileBytes, "invalid WAV RIFF extent");
  StereoWave result;
  uint16_t channels = 0U, blockAlign = 0U;
  uint32_t sampleRate = 0U, byteRate = 0U;
  uint64_t dataOffset = 0U, dataBytes = 0U;
  bool haveFormat = false, haveData = false;
  uint64_t offset = 12U;
  while (offset + 8U <= riffEnd) {
    input.seekg(static_cast<std::streamoff>(offset));
    std::array<uint8_t, 8> chunk{};
    readExact(input, chunk.data(), chunk.size());
    const uint64_t size = little32(chunk.data() + 4U);
    const uint64_t payload = offset + 8U;
    require(size <= riffEnd - payload, "WAV chunk exceeds RIFF extent");
    if (std::memcmp(chunk.data(), "fmt ", 4U) == 0) {
      require(!haveFormat && size >= 16U && size <= 4096U,
              "duplicate or unsupported WAV format chunk");
      std::array<uint8_t, 16> format{};
      readExact(input, format.data(), format.size());
      result.format = little16(format.data());
      channels = little16(format.data() + 2U);
      sampleRate = little32(format.data() + 4U);
      byteRate = little32(format.data() + 8U);
      blockAlign = little16(format.data() + 12U);
      result.bitsPerSample = little16(format.data() + 14U);
      haveFormat = true;
    } else if (std::memcmp(chunk.data(), "data", 4U) == 0) {
      require(!haveData, "multiple WAV data chunks are unsupported");
      dataOffset = payload; dataBytes = size; haveData = true;
    }
    offset = payload + size + (size & 1U);
    require(offset <= riffEnd, "WAV padding exceeds RIFF extent");
  }
  require(haveFormat && haveData && channels == 2U && sampleRate == 44100U &&
              ((result.format == 1U && result.bitsPerSample == 16U) ||
               (result.format == 3U && result.bitsPerSample == 32U)),
          "WAV must be stereo44100 PCM16 or IEEE FLOAT32");
  const uint16_t sampleBytes = result.bitsPerSample / 8U;
  require(blockAlign == 2U * sampleBytes && byteRate == 44100U * blockAlign &&
              dataBytes % blockAlign == 0U, "WAV data alignment differs");
  result.fileFrames = dataBytes / blockAlign;
  result.framesRead = requestedFrames == 0U ? result.fileFrames : requestedFrames;
  require(result.framesRead > 0U && result.framesRead <= result.fileFrames &&
              result.framesRead <= 10000000U, "WAV requested frame count is invalid");
  result.interleaved.resize(static_cast<size_t>(result.framesRead) * 2U);
  input.seekg(static_cast<std::streamoff>(dataOffset));
  std::array<uint8_t, 32768> bytes{};
  uint64_t done = 0U;
  while (done < result.framesRead) {
    const size_t frames = static_cast<size_t>(std::min<uint64_t>(
        result.framesRead - done, bytes.size() / blockAlign));
    readExact(input, bytes.data(), frames * blockAlign);
    for (size_t sample = 0; sample < frames * 2U; ++sample) {
      const uint8_t* value = bytes.data() + sample * sampleBytes;
      float decoded = 0.0F;
      if (result.format == 1U) {
        const uint16_t bits = little16(value);
        const int32_t signedValue = bits < 32768U
            ? static_cast<int32_t>(bits) : static_cast<int32_t>(bits) - 65536;
        decoded = static_cast<float>(signedValue) / 32768.0F;
      } else {
        const uint32_t bits = little32(value);
        std::memcpy(&decoded, &bits, sizeof(decoded));
      }
      require(std::isfinite(decoded), "non-finite input/reference WAV sample");
      result.interleaved[static_cast<size_t>(done) * 2U + sample] = decoded;
    }
    done += frames;
  }
  return result;
}

std::vector<AudioChunk> splitStereoHops(const StereoWave& wave) {
  const size_t hops = static_cast<size_t>((wave.framesRead + kHopSamples - 1U) /
                                        kHopSamples);
  std::vector<AudioChunk> chunks(hops);  // Value initialization pads only final EOF.
  for (uint64_t frame = 0; frame < wave.framesRead; ++frame) {
    const size_t hop = static_cast<size_t>(frame / kHopSamples);
    const size_t offset = static_cast<size_t>(frame % kHopSamples);
    for (size_t channel = 0; channel < kChannels; ++channel)
      chunks[hop][channel * kHopSamples + offset] =
          wave.interleaved[static_cast<size_t>(frame) * kChannels + channel];
  }
  return chunks;
}

struct Summary {
  double mean{};
  double p50{};
  double p95{};
  double p99{};
  double p99_9{};
  double maximum{};
};

double percentile(const std::vector<double>& sorted, double fraction) {
  if (sorted.empty()) return 0.0;
  const double position = fraction * static_cast<double>(sorted.size() - 1U);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double remainder = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - remainder) + sorted[upper] * remainder;
}

Summary summarize(std::vector<double> values) {
  require(!values.empty(), "cannot summarize an empty timing vector");
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  std::sort(values.begin(), values.end());
  return {mean, percentile(values, 0.50), percentile(values, 0.95),
          percentile(values, 0.99), percentile(values, 0.999), values.back()};
}

struct ConsumerSnapshot {
  SeparatedChunk output{};
  bool allIncomingStatesNonzero{};
  AlignedChunk aligned{};
  bool valid{};
  bool sequencePass{};
  bool validityPass{};
  bool alignmentPass{};
  bool pdcPass{};
  bool reconstructionPass{};
  bool nextHistoryPass{};
  bool releasePass{};
  double maximumMixtureError{};
};

ConsumerSnapshot inspectConsumerOutput(
    AsyncQueue& queue, const OutputClaim& claim, uint64_t expectedSequence,
    uint64_t callbackBoundary, const AudioChunk* expectedAligned,
    bool expectedValid) {
  require(claim.slot != nullptr && claim.slot->record != nullptr,
          "consumer claimed an incomplete queue slot");
  Slot& slot = *claim.slot;
  Record& record = *slot.record;
  ConsumerSnapshot snapshot;
  snapshot.output = slot.output;
  snapshot.aligned = slot.aligned;
  snapshot.valid = record.outputValid;
  snapshot.allIncomingStatesNonzero = record.allIncomingStatesNonzero;
  snapshot.sequencePass = slot.sequence == expectedSequence;
  snapshot.validityPass = record.inferenceSucceeded &&
                          record.outputValid == expectedValid;
  const AlignedChunk zero{};
  const AlignedChunk& expected =
      expectedAligned == nullptr ? zero : *expectedAligned;
  snapshot.alignmentPass = slot.aligned == expected;
  if (!expectedValid) {
    snapshot.alignmentPass = snapshot.alignmentPass &&
                             std::all_of(slot.output.begin(), slot.output.end(),
                                         [](float value) {
                                           return value == 0.0F;
                                         });
  }
  snapshot.pdcPass =
      expectedSequence == 0U
          ? callbackBoundary == 1U && !expectedValid
          : callbackBoundary == expectedSequence + 1U && expectedValid &&
                (callbackBoundary - (expectedSequence - 1U)) * kHopSamples == 512U;
  for (size_t channel = 0; channel < static_cast<size_t>(kChannels);
       ++channel) {
    for (size_t sample = 0; sample < static_cast<size_t>(kHopSamples);
         ++sample) {
      float sum = 0.0F;
      for (size_t stem = 0; stem < static_cast<size_t>(kStems); ++stem) {
        const size_t offset =
            (stem * static_cast<size_t>(kChannels) + channel) *
                static_cast<size_t>(kHopSamples) +
            sample;
        sum += slot.output[offset];
      }
      const float mixture =
          slot.aligned[channel * static_cast<size_t>(kHopSamples) + sample];
      snapshot.maximumMixtureError = std::max(
          snapshot.maximumMixtureError,
          std::abs(static_cast<double>(sum) - static_cast<double>(mixture)));
    }
  }
  snapshot.reconstructionPass = snapshot.maximumMixtureError <= 1.0e-6;
  snapshot.nextHistoryPass = record.nextHistoryBitExact &&
                          record.nextHistoryMaximumError == 0.0;
  record.consumerSequencePass = snapshot.sequencePass;
  record.consumerValidityPass = snapshot.validityPass;
  record.consumerAlignmentPass = snapshot.alignmentPass;
  record.consumerPdcPass = snapshot.pdcPass;
  record.consumerReconstructionPass = snapshot.reconstructionPass;
  record.consumerMaximumMixtureError = snapshot.maximumMixtureError;
  snapshot.releasePass = queue.releaseClaim(claim);
  return snapshot;
}

struct StreamProof {
  uint64_t realFrames{};
  uint64_t realInputHops{};
  uint64_t graphCalls{};
  uint64_t graphFlushCalls{};
  uint64_t queueOnlyDrainBoundaries{};
  uint64_t validFramesRecovered{};
  uint64_t startupZeroInvalidHops{};
  bool allConsumerChecks{true};
  bool nonzeroIncomingStatesSeen{};
  bool lastRealSampleRecovered{};
  double maximumMixtureError{};
};

struct PartialEofResult {
  uint64_t frames{};
  StreamProof stream;
  bool passed{};
};

struct ReplayCapture {
  std::array<ConsumerSnapshot, 3> callbacks;
  double resetCallMilliseconds{};
  bool allConsumerChecks{};
  bool oneZeroFlushPass{};
  StreamProof stream;
};

struct InFlightResetResult {
  bool inFlightObserved{};
  double resetCallMilliseconds{};
  bool resetCallNonBlocking{};
  bool staleRunDiscarded{};
  bool oldOutputUnavailable{};
  bool workerAcknowledged{};
};

struct ModeResult {
  Summary workerRun;
  Summary inputToPostPublication;
  Summary callback;
  Summary callbackWakeLateness;
  uint64_t nominalOneHopDeadlineMisses{};
  uint64_t unavailableAtCallbackBoundary{};
  uint64_t availableAtCallbackBoundary{};
  uint64_t actualBoundaryChecksRecorded{};
  uint64_t queueFullDrops{};
  uint64_t staleDiscards{};
  uint64_t inFlightResetDiscards{};
  uint64_t lateTimelineDiscards{};
  uint64_t orderingFailures{};
  uint64_t sequenceGaps{};
  uint64_t resetRequests{};
  uint64_t resetAcknowledgements{};
  uint64_t consumerClaims{};
  uint64_t consumerReleases{};
  size_t prerollInvalidCount{};
  size_t startupZeroInvalidCount{};
  bool graphFlushSubmitted{};
  bool queueOnlyDrainPass{};
  std::vector<PartialEofResult> partialEof;
  std::vector<std::string> inferenceErrors;
  size_t validOutputCount{};
  size_t exactPdcMappingCount{};
  size_t nextHistoryExactCount{};
  double consumerMaximumMixtureError{};
  InFlightResetResult inFlightReset;
  ReplayCapture firstReplay;
  ReplayCapture secondReplay;
  bool deterministicReplayBitExact{};
  double deterministicReplayMaximumDifference{};
  bool correctnessPass{};
  bool diagnosticTimingPass{};
  bool realtimeSchedulingObserved{};
  WorkerPlatform platform;
};

struct AudioBinding {
  std::filesystem::path path;
  std::string sha256;
};

struct Arguments {
  std::filesystem::path model;
  std::filesystem::path ortLibrary;
  std::filesystem::path output;
  std::string modelSha256;
  uintmax_t modelBytes{};
  std::string ortSha256;
  AudioBinding audioInput;
  std::array<AudioBinding, kStems> reference;
  uint64_t prefixFrames{};
  uint64_t captureStart{};
  uint64_t captureEnd{};
  int callbacks{64};
  int warmup{8};
  int threads{1};
  std::vector<int> cpus;
  bool selfTest{};
};

std::vector<int> parseCpus(std::string_view value) {
  std::vector<int> cpus;
  size_t start = 0;
  while (start < value.size()) {
    const size_t comma = value.find(',', start);
    const std::string token(value.substr(
        start, comma == std::string_view::npos ? value.size() - start
                                                : comma - start));
    require(!token.empty(), "empty CPU index in --cpus");
    cpus.push_back(std::stoi(token));
    if (comma == std::string_view::npos) break;
    start = comma + 1U;
  }
  return cpus;
}

bool isSha256(std::string_view value) {
  return value.size() == 64U && std::all_of(value.begin(), value.end(),
      [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

uint64_t parseCount(const std::string& value, const std::string& option) {
  require(!value.empty() && std::all_of(value.begin(), value.end(),
      [](char c) { return c >= '0' && c <= '9'; }),
      option + " requires a nonnegative decimal integer");
  return std::stoull(value);
}

Arguments parseArguments(int argc, char** argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    const auto value = [&]() -> std::string {
      require(index + 1 < argc, "missing value after " + arg);
      return argv[++index];
    };
    if (arg == "--model") {
      arguments.model = value();
    } else if (arg == "--model-sha256") {
      arguments.modelSha256 = value();
    } else if (arg == "--model-bytes") {
      arguments.modelBytes = parseCount(value(), arg);
    } else if (arg == "--ort-library") {
      arguments.ortLibrary = value();
    } else if (arg == "--ort-sha256") {
      arguments.ortSha256 = value();
    } else if (arg == "--audio-input") {
      arguments.audioInput.path = value();
    } else if (arg == "--audio-input-sha256") {
      arguments.audioInput.sha256 = value();
    } else if (arg == "--prefix-frames") {
      arguments.prefixFrames = parseCount(value(), arg);
    } else if (arg == "--capture-start") {
      arguments.captureStart = parseCount(value(), arg);
    } else if (arg == "--capture-end") {
      arguments.captureEnd = parseCount(value(), arg);
    } else if (arg.rfind("--reference-", 0) == 0U) {
      bool known = false;
      for (size_t stem = 0; stem < kStemNames.size(); ++stem) {
        const std::string option = "--reference-" + std::string(kStemNames[stem]);
        if (arg == option) {
          arguments.reference[stem].path = value(); known = true; break;
        }
        if (arg == option + "-sha256") {
          arguments.reference[stem].sha256 = value(); known = true; break;
        }
      }
      require(known, "unknown reference argument: " + arg);
    } else if (arg == "--output") {
      arguments.output = value();
    } else if (arg == "--callbacks") {
      arguments.callbacks = std::stoi(value());
    } else if (arg == "--warmup") {
      arguments.warmup = std::stoi(value());
    } else if (arg == "--threads") {
      arguments.threads = std::stoi(value());
    } else if (arg == "--cpus") {
      arguments.cpus = parseCpus(value());
    } else if (arg == "--self-test") {
      arguments.selfTest = true;
    } else {
      fail("unknown argument: " + arg);
    }
  }
  if (!arguments.selfTest) {
    require(!arguments.model.empty(), "--model is required");
    require(!arguments.ortLibrary.empty(), "--ort-library is required");
    require(!arguments.output.empty(), "--output is required");
    require(isSha256(arguments.modelSha256) && arguments.modelBytes > 0U,
            "actual --model-sha256 and --model-bytes are required");
    require(isSha256(arguments.ortSha256), "actual --ort-sha256 is required");
    const bool anyAudio = !arguments.audioInput.path.empty() ||
        !arguments.audioInput.sha256.empty() || arguments.prefixFrames != 0U ||
        arguments.captureStart != 0U || arguments.captureEnd != 0U ||
        std::any_of(arguments.reference.begin(), arguments.reference.end(),
          [](const AudioBinding& b) { return !b.path.empty() || !b.sha256.empty(); });
    if (anyAudio) {
      require(arguments.audioInput.path.is_absolute() &&
                  isSha256(arguments.audioInput.sha256),
              "music parity needs absolute --audio-input and its actual SHA256");
      require(arguments.prefixFrames > 0U && arguments.prefixFrames <= 10000000U &&
                  arguments.captureStart < arguments.captureEnd &&
                  arguments.captureEnd <= arguments.prefixFrames,
              "music prefix/capture bounds are invalid or exceed10000000 frames");
      for (size_t stem = 0; stem < arguments.reference.size(); ++stem) {
        const auto& binding = arguments.reference[stem];
        require(binding.path.is_absolute() && isSha256(binding.sha256),
                "all four absolute reference WAV paths and hashes are required");
        require(binding.path != arguments.audioInput.path,
                "reference WAV must differ from the mixture");
        for (size_t earlier = 0; earlier < stem; ++earlier)
          require(binding.path != arguments.reference[earlier].path,
                  "reference stems must use distinct files");
      }
    }
    require(arguments.callbacks >= 8 && arguments.callbacks <= 100000,
            "--callbacks must be in [8,100000]");
    require(arguments.warmup >= 2 && arguments.warmup <= 1000,
            "--warmup must be in [2,1000]");
    require(arguments.threads >= 1 && arguments.threads <= 16,
            "--threads must be in [1,16]");
  }
  return arguments;
}

void waitRecord(AsyncQueue& queue, Record& record,
                std::chrono::milliseconds timeout) {
  const auto deadline = Clock::now() + timeout;
  while (!record.workerFinished.load(std::memory_order_acquire)) {
    queue.rethrowWorkerFailure();
    require(Clock::now() < deadline, "inference record timed out");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

std::optional<OutputClaim> waitForExpectedClaim(
    AsyncQueue& queue, uint32_t epoch, uint64_t sequence,
    std::chrono::milliseconds timeout) {
  const TimePoint deadline = Clock::now() + timeout;
  while (Clock::now() < deadline) {
    if (auto claim = queue.claimExpected(epoch, sequence)) return claim;
    queue.rethrowWorkerFailure();
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return std::nullopt;
}

bool consumerChecksPass(const ConsumerSnapshot& snapshot) {
  return snapshot.sequencePass && snapshot.validityPass &&
         snapshot.alignmentPass && snapshot.pdcPass &&
         snapshot.reconstructionPass && snapshot.nextHistoryPass &&
         snapshot.releasePass;
}

template <typename Consumer>
StreamProof runBoundedStream(AsyncQueue& queue,
                            const std::vector<AudioChunk>& inputs,
                            uint64_t realFrames, bool resetFirst,
                            Consumer&& consume) {
  require(!inputs.empty() && realFrames > 0U &&
              inputs.size() == (realFrames + kHopSamples - 1U) / kHopSamples,
          "bounded stream frame/hop geometry differs");
  if (resetFirst) {
    const ResetEvent reset = queue.resetNonBlocking();
    queue.waitForResetAcknowledgement(reset.epoch);
  }
  const uint32_t epoch = queue.currentEpoch();
  StreamProof proof;
  proof.realFrames = realFrames;
  proof.realInputHops = inputs.size();
  // This untimed functional pass uses the actual queue and callback ordering.
  // Waiting between boundaries is deliberate and is not a deadline measurement.
  // Boundary0 publishes a zero, invalid output before the first input is queued.
  const auto unexpected = queue.claimExpected(epoch, 0U);
  if (unexpected) static_cast<void>(queue.releaseClaim(*unexpected));
  require(!unexpected.has_value(), "reset epoch has a stale startup output");
  const SeparatedChunk firstPublication{};
  require(std::all_of(firstPublication.begin(), firstPublication.end(),
                     [](float value) { return value == 0.0F; }),
          "first queue publication must be zero and invalid");
  proof.startupZeroInvalidHops = 1U;
  const AudioChunk zero{};
  std::vector<Record> records(inputs.size() + 1U);
  const StopWorkerOnUnwind recordsLifetime(queue);
  for (size_t boundary = 0; boundary <= records.size(); ++boundary) {
    if (boundary > 0U) {
      const size_t due = boundary - 1U;
      const auto claim = queue.claimExpected(epoch, due);
      require(claim.has_value(), "exact bounded-stream due output is unavailable");
      const AudioChunk* expected = due == 0U ? nullptr : &inputs[due - 1U];
      const auto snapshot = inspectConsumerOutput(queue, *claim, due, boundary,
                                                   expected, due != 0U);
      proof.allConsumerChecks = proof.allConsumerChecks &&
                                consumerChecksPass(snapshot);
      proof.nonzeroIncomingStatesSeen = proof.nonzeroIncomingStatesSeen ||
                                        snapshot.allIncomingStatesNonzero;
      proof.maximumMixtureError = std::max(proof.maximumMixtureError,
                                           snapshot.maximumMixtureError);
      if (due == 0U && !snapshot.valid &&
          std::all_of(snapshot.output.begin(), snapshot.output.end(),
                      [](float value) { return value == 0.0F; }))
        ++proof.startupZeroInvalidHops;
      uint64_t physicalStart = 0U, validFrames = 0U;
      if (due > 0U) {
        physicalStart = static_cast<uint64_t>(due - 1U) * kHopSamples;
        validFrames = std::min<uint64_t>(kHopSamples, realFrames - physicalStart);
        if (snapshot.valid) proof.validFramesRecovered += validFrames;
        if (physicalStart + validFrames == realFrames && snapshot.valid) {
          const size_t last = static_cast<size_t>(validFrames - 1U);
          proof.lastRealSampleRecovered = true;
          for (size_t channel = 0; channel < kChannels; ++channel)
            proof.lastRealSampleRecovered = proof.lastRealSampleRecovered &&
                snapshot.aligned[channel * kHopSamples + last] ==
                    inputs.back()[channel * kHopSamples + last];
        }
      }
      consume(due, snapshot, physicalStart, validFrames);
      if (boundary == records.size()) ++proof.queueOnlyDrainBoundaries;
    }
    if (boundary < records.size()) {
      const bool flush = boundary == inputs.size();
      const AudioChunk& input = flush ? zero : inputs[boundary];
      require(queue.submit(input, boundary, epoch, records[boundary]),
              "bounded-stream queue submission failed");
      ++proof.graphCalls;
      if (flush) ++proof.graphFlushCalls;
      waitRecord(queue, records[boundary], std::chrono::seconds(2));
      require(records[boundary].inferenceSucceeded,
              "bounded-stream inference failed at sequence " +
                  std::to_string(boundary) + ": " + records[boundary].inferenceError);
    }
  }
  proof.allConsumerChecks = proof.allConsumerChecks &&
      proof.startupZeroInvalidHops == 2U &&
      proof.graphCalls == inputs.size() + 1U && proof.graphFlushCalls == 1U &&
      proof.queueOnlyDrainBoundaries == 1U &&
      proof.validFramesRecovered == realFrames && proof.lastRealSampleRecovered;
  return proof;
}

ReplayCapture runReplay(AsyncQueue& queue,
                        const std::vector<AudioChunk>& inputs,
                        bool resetFirst) {
  require(inputs.size() >= 2U, "replay requires two deterministic inputs");
  ReplayCapture replay;
  if (resetFirst) {
    const ResetEvent reset = queue.resetNonBlocking();
    replay.resetCallMilliseconds = reset.callMilliseconds;
    queue.waitForResetAcknowledgement(reset.epoch);
  }
  const std::vector<AudioChunk> excerpt = {inputs[0], inputs[1]};
  replay.stream = runBoundedStream(queue, excerpt, 2U * kHopSamples, false,
      [&](size_t sequence, const ConsumerSnapshot& snapshot, uint64_t, uint64_t) {
        require(sequence < replay.callbacks.size(), "replay capture index differs");
        replay.callbacks[sequence] = snapshot;
      });
  replay.allConsumerChecks = replay.stream.allConsumerChecks;
  replay.oneZeroFlushPass = replay.stream.graphFlushCalls == 1U &&
      replay.stream.queueOnlyDrainBoundaries == 1U &&
      replay.callbacks[2].valid && replay.callbacks[2].aligned == inputs[1];
  return replay;
}

std::vector<PartialEofResult> runPartialEofFixtures(AsyncQueue& queue) {
  std::vector<PartialEofResult> results;
  for (const uint64_t frames : {1U, 255U, 256U, 257U, 511U, 512U, 513U, 769U}) {
    const size_t hops = static_cast<size_t>((frames + kHopSamples - 1U) / kHopSamples);
    std::vector<AudioChunk> input(hops);
    for (size_t hop = 0; hop < hops; ++hop) {
      input[hop] = makeAudioChunk(static_cast<int64_t>(hop * kHopSamples));
      for (size_t sample = 0; sample < kHopSamples; ++sample)
        if (hop * kHopSamples + sample >= frames)
          for (size_t channel = 0; channel < kChannels; ++channel)
            input[hop][channel * kHopSamples + sample] = 0.0F;
    }
    const size_t last = static_cast<size_t>((frames - 1U) % kHopSamples);
    input.back()[last] = 0.625F;
    input.back()[kHopSamples + last] = -0.375F;
    PartialEofResult result;
    result.frames = frames;
    bool lastSentinelPass = false;
    result.stream = runBoundedStream(queue, input, frames, true,
        [&](size_t sequence, const ConsumerSnapshot& snapshot,
            uint64_t first, uint64_t valid) {
          if (sequence > 0U && first + valid == frames)
            lastSentinelPass = snapshot.valid && snapshot.aligned[last] == 0.625F &&
                              snapshot.aligned[kHopSamples + last] == -0.375F;
        });
    result.passed = result.stream.allConsumerChecks && lastSentinelPass;
    results.push_back(result);
  }
  return results;
}

double replayMaximumDifference(const ReplayCapture& lhs,
                               const ReplayCapture& rhs) {
  double maximum = 0.0;
  for (size_t callback = 0; callback < lhs.callbacks.size(); ++callback) {
    for (size_t index = 0; index < lhs.callbacks[callback].output.size();
         ++index) {
      maximum = std::max(
          maximum,
          std::abs(static_cast<double>(lhs.callbacks[callback].output[index]) -
                   static_cast<double>(rhs.callbacks[callback].output[index])));
    }
    for (size_t index = 0; index < lhs.callbacks[callback].aligned.size();
         ++index) {
      maximum = std::max(
          maximum,
          std::abs(static_cast<double>(lhs.callbacks[callback].aligned[index]) -
                   static_cast<double>(rhs.callbacks[callback].aligned[index])));
    }
  }
  return maximum;
}

bool replayBitExact(const ReplayCapture& lhs, const ReplayCapture& rhs) {
  for (size_t callback = 0; callback < lhs.callbacks.size(); ++callback) {
    if (lhs.callbacks[callback].valid != rhs.callbacks[callback].valid ||
        lhs.callbacks[callback].output != rhs.callbacks[callback].output ||
        lhs.callbacks[callback].aligned != rhs.callbacks[callback].aligned) {
      return false;
    }
  }
  return true;
}

InFlightResetResult runInFlightResetTest(
    AsyncQueue& queue, const AudioChunk& input) {
  InFlightResetResult result;
  const ResetEvent preparation = queue.resetNonBlocking();
  queue.waitForResetAcknowledgement(preparation.epoch);
  const uint32_t oldEpoch = queue.currentEpoch();
  Record oldRecord;
  const StopWorkerOnUnwind recordLifetime(queue);
  require(queue.submit(input, 0U, oldEpoch, oldRecord),
          "in-flight reset probe submission failed");
  const TimePoint observeDeadline = Clock::now() + std::chrono::seconds(2);
  while (Clock::now() < observeDeadline) {
    if (queue.workerInFlightSequence() == 0U) {
      result.inFlightObserved = true;
      break;
    }
    if (oldRecord.workerFinished.load(std::memory_order_acquire)) break;
    std::this_thread::yield();
  }
  const ResetEvent reset = queue.resetNonBlocking();
  result.resetCallMilliseconds = reset.callMilliseconds;
  result.resetCallNonBlocking = reset.callMilliseconds < 1.0;
  waitRecord(queue, oldRecord, std::chrono::seconds(2));
  queue.waitForResetAcknowledgement(reset.epoch);
  result.workerAcknowledged =
      queue.currentEpoch() == reset.epoch &&
      queue.workerAcknowledgedEpoch() >= reset.epoch;
  result.staleRunDiscarded = oldRecord.staleDiscarded &&
                             oldRecord.postPublicationNanoseconds.load(
                                 std::memory_order_acquire) == 0;
  const auto oldClaim = queue.claimExpected(oldEpoch, 0U);
  result.oldOutputUnavailable = !oldClaim.has_value();
  if (oldClaim) static_cast<void>(queue.releaseClaim(*oldClaim));
  return result;
}

ModeResult runMode(const Arguments& arguments,
                   const std::vector<AudioChunk>& inputs) {
  AsyncQueue queue(arguments.model, arguments.threads, arguments.cpus);
  queue.start();

  for (int warmup = 0; warmup < arguments.warmup; ++warmup) {
    Record record;
    const StopWorkerOnUnwind recordLifetime(queue);
    record.scheduledCallback = Clock::now();
    record.deadline = record.scheduledCallback +
                      std::chrono::duration_cast<Clock::duration>(
                          std::chrono::duration<double, std::milli>(
                              kDeadlineMilliseconds));
    require(queue.submit(inputs[static_cast<size_t>(warmup) % inputs.size()],
                         static_cast<uint64_t>(warmup), queue.currentEpoch(),
                         record),
            "warmup queue submission failed");
    waitRecord(queue, record, std::chrono::seconds(2));
    require(record.inferenceSucceeded,
            "warmup inference failed: " + record.inferenceError);
    const auto claim = waitForExpectedClaim(
        queue, queue.currentEpoch(), static_cast<uint64_t>(warmup),
        std::chrono::seconds(2));
    require(claim.has_value(), "warmup output was unavailable");
    const AudioChunk* expectedAligned =
        warmup == 0
            ? nullptr
            : &inputs[static_cast<size_t>(warmup - 1) % inputs.size()];
    static_cast<void>(inspectConsumerOutput(
        queue, *claim, static_cast<uint64_t>(warmup),
        static_cast<uint64_t>(warmup + 1), expectedAligned, warmup != 0));
  }

  ModeResult result;
  result.inFlightReset = runInFlightResetTest(queue, inputs[0]);
  // The first replay starts immediately in the epoch installed while the old
  // graph call was in flight.  The second replay follows a fresh idle reset.
  // Comparing them proves that discarding the in-flight result did not leak
  // recurrent state into the new epoch.
  result.firstReplay = runReplay(queue, inputs, false);
  result.secondReplay = runReplay(queue, inputs, true);
  result.deterministicReplayMaximumDifference =
      replayMaximumDifference(result.firstReplay, result.secondReplay);
  result.deterministicReplayBitExact =
      replayBitExact(result.firstReplay, result.secondReplay) &&
      result.deterministicReplayMaximumDifference == 0.0;

  result.partialEof = runPartialEofFixtures(queue);

  const ResetEvent pacedReset = queue.resetNonBlocking();
  queue.waitForResetAcknowledgement(pacedReset.epoch);
  const uint32_t pacedEpoch = queue.currentEpoch();

  // N genuine inputs + exactly one graph flush. The final boundary only drains.
  std::vector<Record> records(static_cast<size_t>(arguments.callbacks) + 1U);
  const StopWorkerOnUnwind recordsLifetime(queue);
  std::vector<double> callbackMilliseconds;
  std::vector<double> callbackWakeMilliseconds;
  callbackMilliseconds.reserve(records.size() + 1U);
  callbackWakeMilliseconds.reserve(records.size() + 1U);
  const auto hopDuration = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(static_cast<double>(kHopSamples) /
                                    static_cast<double>(kSampleRate)));
  const TimePoint base = Clock::now() + std::chrono::milliseconds(20);
  std::this_thread::sleep_until(base);
  const AudioChunk pacedZero{};
  bool queueStartupEmpty = true;
  for (int callback = 0; callback <= arguments.callbacks + 1; ++callback) {
    const TimePoint scheduled = base + callback * hopDuration;
    if (callback > 0) std::this_thread::sleep_until(scheduled);
    const TimePoint callbackBegin = Clock::now();
    callbackWakeMilliseconds.push_back(std::max(
        0.0, std::chrono::duration<double, std::milli>(callbackBegin - scheduled)
                 .count()));
    ConsumerSnapshot publication;  // Missing/delayed output publishes zero, invalid.
    if (callback == 0) {
      const auto unexpected = queue.claimExpected(pacedEpoch, 0U);
      queueStartupEmpty = !unexpected.has_value();
      if (unexpected) static_cast<void>(queue.releaseClaim(*unexpected));
    }
    if (callback > 0) {
      const size_t due = static_cast<size_t>(callback - 1);
      Record& dueRecord = records[due];
      dueRecord.callbackChecked = true;
      const auto claim = queue.claimExpected(pacedEpoch, due);
      dueRecord.consumerBoundaryObservationNanoseconds.store(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              Clock::now().time_since_epoch())
              .count(),
          std::memory_order_release);
      dueRecord.availableAtActualBoundary = claim.has_value();
      if (claim) {
        const AudioChunk* expectedAligned =
            due == 0U ? nullptr : &inputs[(due - 1U) % inputs.size()];
        publication = inspectConsumerOutput(
            queue, *claim, due, static_cast<uint64_t>(callback),
            expectedAligned, due != 0U);
      }
    }
    if (callback < 2 && !publication.valid &&
        std::all_of(publication.output.begin(), publication.output.end(),
                    [](float value) { return value == 0.0F; }) &&
        (callback == 0 ? queueStartupEmpty : publication.validityPass))
      ++result.startupZeroInvalidCount;
    if (callback == arguments.callbacks + 1)
      result.queueOnlyDrainPass = consumerChecksPass(publication) &&
          publication.valid && publication.aligned ==
              inputs[static_cast<size_t>(arguments.callbacks - 1) % inputs.size()];
    if (callback < static_cast<int>(records.size())) {
      Record& record = records[static_cast<size_t>(callback)];
      record.scheduledCallback = scheduled;
      record.deadline = scheduled + hopDuration;
      const bool flush = callback == arguments.callbacks;
      const bool submitted = queue.submit(
          flush ? pacedZero : inputs[static_cast<size_t>(callback) % inputs.size()],
          static_cast<uint64_t>(callback), pacedEpoch, record);
      if (flush) result.graphFlushSubmitted = submitted;
    }
    const TimePoint callbackEnd = Clock::now();
    callbackMilliseconds.push_back(
        std::chrono::duration<double, std::milli>(callbackEnd - callbackBegin)
            .count());
  }

  for (Record& record : records) {
    if (record.submitted &&
        !record.workerFinished.load(std::memory_order_acquire)) {
      waitRecord(queue, record, std::chrono::seconds(2));
    }
  }

  std::vector<double> workerMilliseconds;
  std::vector<double> inputToPostPublicationMilliseconds;
  workerMilliseconds.reserve(records.size());
  inputToPostPublicationMilliseconds.reserve(records.size());
  result.callback = summarize(callbackMilliseconds);
  result.callbackWakeLateness = summarize(callbackWakeMilliseconds);
  for (size_t index = 0; index < records.size(); ++index) {
    const Record& record = records[index];
    if (!record.inferenceError.empty())
      result.inferenceErrors.push_back("sequence " + std::to_string(index) +
                                       ": " + record.inferenceError);
    if (!record.submitted || record.staleDiscarded) continue;
    workerMilliseconds.push_back(
        std::chrono::duration<double, std::milli>(record.runEnd -
                                                  record.runBegin)
            .count());
    const int64_t postPublication =
        record.postPublicationNanoseconds.load(std::memory_order_acquire);
    const int64_t inputPublication =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            record.inputPublished.time_since_epoch())
            .count();
    inputToPostPublicationMilliseconds.push_back(
        static_cast<double>(postPublication - inputPublication) / 1.0e6);
    const int64_t nominalDeadline =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            record.deadline.time_since_epoch())
            .count();
    if (postPublication >= nominalDeadline) {
      ++result.nominalOneHopDeadlineMisses;
    }
    if (record.callbackChecked && record.availableAtActualBoundary) {
      ++result.availableAtCallbackBoundary;
    } else {
      ++result.unavailableAtCallbackBoundary;
    }
    if (record.callbackChecked &&
        record.consumerBoundaryObservationNanoseconds.load(
            std::memory_order_acquire) != 0) {
      ++result.actualBoundaryChecksRecorded;
    }
    if (record.outputValid) {
      ++result.validOutputCount;
    } else {
      ++result.prerollInvalidCount;
    }
    if (record.consumerPdcPass) ++result.exactPdcMappingCount;
    if (record.nextHistoryBitExact && record.nextHistoryMaximumError == 0.0) {
      ++result.nextHistoryExactCount;
    }
    result.consumerMaximumMixtureError = std::max(
        result.consumerMaximumMixtureError,
        record.consumerMaximumMixtureError);
  }
  result.workerRun = summarize(workerMilliseconds);
  result.inputToPostPublication =
      summarize(inputToPostPublicationMilliseconds);
  result.queueFullDrops =
      queue.telemetry().queueFullDrops.load(std::memory_order_acquire);
  result.staleDiscards =
      queue.telemetry().staleDiscards.load(std::memory_order_acquire);
  result.inFlightResetDiscards =
      queue.telemetry().inFlightResetDiscards.load(std::memory_order_acquire);
  result.lateTimelineDiscards =
      queue.telemetry().lateTimelineDiscards.load(std::memory_order_acquire);
  result.orderingFailures =
      queue.telemetry().orderingFailures.load(std::memory_order_acquire);
  result.sequenceGaps =
      queue.telemetry().sequenceGaps.load(std::memory_order_acquire);
  result.resetRequests =
      queue.telemetry().resetRequests.load(std::memory_order_acquire);
  result.resetAcknowledgements =
      queue.telemetry().resetAcknowledgements.load(std::memory_order_acquire);
  result.consumerClaims =
      queue.telemetry().consumerClaims.load(std::memory_order_acquire);
  result.consumerReleases =
      queue.telemetry().consumerReleases.load(std::memory_order_acquire);
  result.platform = queue.platform();
  const bool timedConsumerChecks = std::all_of(
      records.begin(), records.end(), [](const Record& record) {
        return record.callbackChecked && record.availableAtActualBoundary &&
               record.consumerSequencePass && record.consumerValidityPass &&
               record.consumerAlignmentPass && record.consumerPdcPass &&
               record.consumerReconstructionPass && record.nextHistoryBitExact &&
               record.nextHistoryMaximumError == 0.0;
      });
  result.correctnessPass =
      result.inFlightReset.inFlightObserved &&
      result.inFlightReset.resetCallNonBlocking &&
      result.inFlightReset.staleRunDiscarded &&
      result.inFlightReset.oldOutputUnavailable &&
      result.inFlightReset.workerAcknowledged &&
      result.firstReplay.allConsumerChecks &&
      result.secondReplay.allConsumerChecks &&
      result.firstReplay.oneZeroFlushPass &&
      result.secondReplay.oneZeroFlushPass &&
      result.deterministicReplayBitExact && timedConsumerChecks &&
      result.inferenceErrors.empty() &&
      result.partialEof.size() == 8U &&
      std::all_of(result.partialEof.begin(), result.partialEof.end(),
                  [](const PartialEofResult& row) { return row.passed; }) &&
      result.startupZeroInvalidCount == 2U && result.graphFlushSubmitted &&
      result.queueOnlyDrainPass &&
      result.validOutputCount == static_cast<size_t>(arguments.callbacks) &&
      result.actualBoundaryChecksRecorded == records.size() &&
      result.prerollInvalidCount == 1U &&
      result.validOutputCount + result.prerollInvalidCount == records.size() &&
      result.exactPdcMappingCount == records.size() &&
      result.nextHistoryExactCount == records.size() &&
      result.queueFullDrops == 0U && result.orderingFailures == 0U &&
      result.sequenceGaps == 0U &&
      result.resetRequests == result.resetAcknowledgements &&
      result.inFlightResetDiscards >= 1U &&
      result.consumerMaximumMixtureError <= 1.0e-6;
  result.diagnosticTimingPass =
      result.nominalOneHopDeadlineMisses == 0U &&
      result.unavailableAtCallbackBoundary == 0U &&
      result.queueFullDrops == 0U && result.orderingFailures == 0U;
  result.realtimeSchedulingObserved =
      (result.platform.schedulingPolicy == SCHED_RR ||
       result.platform.schedulingPolicy == SCHED_FIFO) &&
      result.platform.schedulingPriority > 0;
  return result;
}

struct MusicParityResult {
  bool executed{};
  bool passed{};
  bool audioInputsUnchanged{};
  uint16_t inputFormat{};
  uint16_t inputBitsPerSample{};
  uint64_t inputFileFrames{};
  uint64_t capturedFrames{};
  double elapsedSeconds{};
  StreamProof stream;
  std::array<double, kStems> maximumAbsoluteError{};
  std::array<double, kStems> maximumCallbackRmsError{};
  std::array<double, kStems> overallRmsError{};
};

MusicParityResult runMusicParity(const Arguments& arguments) {
  MusicParityResult result;
  if (arguments.audioInput.path.empty()) return result;
  result.executed = true;
  const auto authenticate = [](const AudioBinding& binding) {
    require(std::filesystem::is_regular_file(binding.path) &&
                sha256File(binding.path) == binding.sha256,
            "actual audio/reference identity changed: " + binding.path.string());
  };
  authenticate(arguments.audioInput);
  for (const auto& binding : arguments.reference) authenticate(binding);
  const StereoWave mixture = readStereoWave(arguments.audioInput.path,
                                              arguments.prefixFrames);
  std::array<StereoWave, kStems> references;
  const uint64_t referenceFrames = arguments.captureEnd - arguments.captureStart;
  for (size_t stem = 0; stem < references.size(); ++stem) {
    references[stem] = readStereoWave(arguments.reference[stem].path);
    require(references[stem].format == 3U && references[stem].bitsPerSample == 32U &&
                references[stem].fileFrames == referenceFrames,
            "saved stem reference must be unnormalized FLOAT32 with exact capture length");
  }
  result.inputFormat = mixture.format;
  result.inputBitsPerSample = mixture.bitsPerSample;
  result.inputFileFrames = mixture.fileFrames;
  const std::vector<AudioChunk> inputs = splitStereoHops(mixture);
  std::array<double, kStems> sumSquaredError{};
  AsyncQueue queue(arguments.model, arguments.threads, arguments.cpus);
  queue.start();
  const TimePoint began = Clock::now();
  // Continuous genuine prefix from physical0. No interior crop, reset or flush.
  // The handler receives physical offsets after graph256 + queue256; capture
  // boundaries therefore correspond to callback sample positions physical+512.
  result.stream = runBoundedStream(queue, inputs, arguments.prefixFrames, false,
      [&](size_t sequence, const ConsumerSnapshot& snapshot,
          uint64_t physicalStart, uint64_t validFrames) {
        if (sequence > 0U && sequence % 2048U == 0U)
          std::cerr << "music parity physical frame " << physicalStart << '\n';
        if (sequence == 0U || !snapshot.valid) return;
        const uint64_t first = std::max(physicalStart, arguments.captureStart);
        const uint64_t end = std::min(physicalStart + validFrames, arguments.captureEnd);
        if (first >= end) return;
        std::array<double, kStems> callbackSquaredError{};
        for (uint64_t physical = first; physical < end; ++physical) {
          const size_t local = static_cast<size_t>(physical - physicalStart);
          const size_t reference = static_cast<size_t>(physical - arguments.captureStart);
          for (size_t stem = 0; stem < kStems; ++stem)
            for (size_t channel = 0; channel < kChannels; ++channel) {
              const double value = snapshot.output[
                  (stem * kChannels + channel) * kHopSamples + local];
              const double wanted = references[stem].interleaved[
                  reference * kChannels + channel];
              const double error = value - wanted;
              result.maximumAbsoluteError[stem] = std::max(
                  result.maximumAbsoluteError[stem], std::abs(error));
              callbackSquaredError[stem] += error * error;
              sumSquaredError[stem] += error * error;
            }
        }
        for (size_t stem = 0; stem < kStems; ++stem)
          result.maximumCallbackRmsError[stem] = std::max(
              result.maximumCallbackRmsError[stem],
              std::sqrt(callbackSquaredError[stem] /
                        static_cast<double>((end - first) * kChannels)));
        result.capturedFrames += end - first;
      });
  result.elapsedSeconds = std::chrono::duration<double>(Clock::now() - began).count();
  queue.stop();
  authenticate(arguments.audioInput);
  for (const auto& binding : arguments.reference) authenticate(binding);
  result.audioInputsUnchanged = true;
  result.passed = result.stream.allConsumerChecks &&
      result.stream.nonzeroIncomingStatesSeen && result.capturedFrames == referenceFrames;
  for (size_t stem = 0; stem < kStems; ++stem) {
    result.overallRmsError[stem] = std::sqrt(sumSquaredError[stem] /
        static_cast<double>(referenceFrames * kChannels));
    result.passed = result.passed &&
        result.maximumAbsoluteError[stem] <= kParityMaximumAbsoluteError &&
        result.maximumCallbackRmsError[stem] <= kParityMaximumCallbackRmsError;
  }
  return result;
}

void writeStreamProof(std::ostream& output, const StreamProof& proof) {
  output << "{\"real_frames\":" << proof.realFrames
         << ",\"real_input_hops\":" << proof.realInputHops
         << ",\"graph_calls\":" << proof.graphCalls
         << ",\"graph_flush_calls\":" << proof.graphFlushCalls
         << ",\"queue_only_drain_boundaries\":" << proof.queueOnlyDrainBoundaries
         << ",\"valid_frames_recovered\":" << proof.validFramesRecovered
         << ",\"startup_zero_invalid_hops\":" << proof.startupZeroInvalidHops
         << ",\"all_consumer_checks\":" << (proof.allConsumerChecks ? "true" : "false")
         << ",\"all_four_incoming_states_observed_nonzero\":"
         << (proof.nonzeroIncomingStatesSeen ? "true" : "false")
         << ",\"last_real_sample_recovered\":"
         << (proof.lastRealSampleRecovered ? "true" : "false")
         << ",\"maximum_mixture_error\":" << proof.maximumMixtureError << '}';
}

void writeMusicParity(std::ostream& output, const Arguments& arguments,
                      const MusicParityResult& result) {
  output << "{\"executed\":" << (result.executed ? "true" : "false")
         << ",\"passed\":" << (result.passed ? "true" : "false")
         << ",\"deadline_measurement\":false,\"output_audio_files_written\":0"
         << ",\"prefix_frames\":" << arguments.prefixFrames
         << ",\"capture_physical_start\":" << arguments.captureStart
         << ",\"capture_physical_end\":" << arguments.captureEnd
         << ",\"capture_callback_start\":" << arguments.captureStart + 512U
         << ",\"capture_callback_end\":" << arguments.captureEnd + 512U
         << ",\"captured_frames\":" << result.capturedFrames
         << ",\"input_file_frames\":" << result.inputFileFrames
         << ",\"input_format_tag\":" << result.inputFormat
         << ",\"input_bits_per_sample\":" << result.inputBitsPerSample
         << ",\"pcm16_decode_divisor\":32768,\"interior_flushes\":0"
         << ",\"elapsed_seconds\":" << result.elapsedSeconds
         << ",\"inputs_unchanged\":" << (result.audioInputsUnchanged ? "true" : "false")
         << ",\"maximum_absolute_error_tolerance\":" << kParityMaximumAbsoluteError
         << ",\"maximum_callback_rms_error_tolerance\":" << kParityMaximumCallbackRmsError
         << ",\"mixture\":{\"path\":\"" << jsonEscape(arguments.audioInput.path.string())
         << "\",\"sha256\":\"" << arguments.audioInput.sha256 << "\"}"
         << ",\"stream\":";
  writeStreamProof(output, result.stream);
  output << ",\"stems\":{";
  for (size_t stem = 0; stem < kStems; ++stem) {
    if (stem != 0U) output << ',';
    output << '\"' << kStemNames[stem] << "\":{\"reference_path\":\""
           << jsonEscape(arguments.reference[stem].path.string())
           << "\",\"reference_sha256\":\"" << arguments.reference[stem].sha256
           << "\",\"maximum_absolute_error\":" << result.maximumAbsoluteError[stem]
           << ",\"maximum_callback_rms_error\":" << result.maximumCallbackRmsError[stem]
           << ",\"overall_rms_error\":" << result.overallRmsError[stem] << '}';
  }
  output << "}}";
}

void writeSummary(std::ostream& output, const Summary& summary) {
  output << "{\"mean\":" << summary.mean << ",\"p50\":" << summary.p50
         << ",\"p95\":" << summary.p95 << ",\"p99\":" << summary.p99
         << ",\"p99_9\":" << summary.p99_9
         << ",\"max\":" << summary.maximum << '}';
}

void writeMode(std::ostream& output, const ModeResult& result) {
  output << "{\"queue_protocol\":\"" << queueModeName()
         << "\",\"worker_run_ms\":";
  writeSummary(output, result.workerRun);
  output << ",\"input_publication_to_post_release_publication_ms\":";
  writeSummary(output, result.inputToPostPublication);
  output << ",\"callback_duration_ms\":";
  writeSummary(output, result.callback);
  output << ",\"callback_wake_lateness_ms\":";
  writeSummary(output, result.callbackWakeLateness);
  output << ",\"nominal_one_hop_deadline_misses\":"
         << result.nominalOneHopDeadlineMisses
         << ",\"available_at_actual_callback_boundary\":"
         << result.availableAtCallbackBoundary
         << ",\"unavailable_at_actual_callback_boundary\":"
         << result.unavailableAtCallbackBoundary
         << ",\"actual_boundary_checks_recorded\":"
         << result.actualBoundaryChecksRecorded
         << ",\"queue_telemetry\":{\"queue_full_drops\":"
         << result.queueFullDrops << ",\"stale_discards\":"
         << result.staleDiscards << ",\"in_flight_reset_discards\":"
         << result.inFlightResetDiscards
         << ",\"late_timeline_discards\":"
         << result.lateTimelineDiscards << ",\"ordering_failures\":"
         << result.orderingFailures << ",\"sequence_gaps\":"
         << result.sequenceGaps << ",\"reset_requests\":"
         << result.resetRequests << ",\"reset_acknowledgements\":"
         << result.resetAcknowledgements << ",\"consumer_claims\":"
         << result.consumerClaims << ",\"consumer_releases\":"
         << result.consumerReleases << "},\"streaming_correctness\":{"
         << "\"preroll_invalid_count\":" << result.prerollInvalidCount
         << ",\"startup_zero_invalid_hops\":" << result.startupZeroInvalidCount
         << ",\"exactly_one_graph_flush_submitted\":"
         << (result.graphFlushSubmitted ? "true" : "false")
         << ",\"final_queue_only_drain_pass\":"
         << (result.queueOnlyDrainPass ? "true" : "false")
         << ",\"valid_output_count\":" << result.validOutputCount
         << ",\"exact_512_pdc_mapping_count\":"
         << result.exactPdcMappingCount
         << ",\"next_history_shift_append_bit_exact_count\":"
         << result.nextHistoryExactCount
         << ",\"consumer_maximum_mixture_reconstruction_error\":"
         << result.consumerMaximumMixtureError
         << ",\"deterministic_replay_bit_exact\":"
         << (result.deterministicReplayBitExact ? "true" : "false")
         << ",\"post_in_flight_reset_replay_matches_fresh_reset\":"
         << (result.deterministicReplayBitExact ? "true" : "false")
         << ",\"deterministic_replay_maximum_difference\":"
         << result.deterministicReplayMaximumDifference
         << ",\"first_replay_consumer_checks\":"
         << (result.firstReplay.allConsumerChecks ? "true" : "false")
         << ",\"second_replay_consumer_checks\":"
         << (result.secondReplay.allConsumerChecks ? "true" : "false")
         << ",\"first_replay_one_zero_flush\":"
         << (result.firstReplay.oneZeroFlushPass ? "true" : "false")
         << ",\"second_replay_one_zero_flush\":"
         << (result.secondReplay.oneZeroFlushPass ? "true" : "false")
         << ",\"first_replay_stream\":";
  writeStreamProof(output, result.firstReplay.stream);
  output << ",\"second_replay_stream\":";
  writeStreamProof(output, result.secondReplay.stream);
  output << ",\"partial_eof_cases\":[";
  for (size_t index = 0; index < result.partialEof.size(); ++index) {
    if (index != 0U) output << ',';
    const auto& row = result.partialEof[index];
    output << "{\"frames\":" << row.frames << ",\"passed\":"
           << (row.passed ? "true" : "false") << ",\"stream\":";
    writeStreamProof(output, row.stream);
    output << '}';
  }
  output << "],\"inference_errors\":[";
  for (size_t index = 0; index < result.inferenceErrors.size(); ++index) {
    if (index != 0U) output << ',';
    output << '\"' << jsonEscape(result.inferenceErrors[index]) << '\"';
  }
  output << "]},\"in_flight_nonblocking_reset\":{\"in_flight_observed\":"
         << (result.inFlightReset.inFlightObserved ? "true" : "false")
         << ",\"reset_call_ms\":"
         << result.inFlightReset.resetCallMilliseconds
         << ",\"reset_call_nonblocking\":"
         << (result.inFlightReset.resetCallNonBlocking ? "true" : "false")
         << ",\"stale_run_discarded\":"
         << (result.inFlightReset.staleRunDiscarded ? "true" : "false")
         << ",\"old_output_unavailable\":"
         << (result.inFlightReset.oldOutputUnavailable ? "true" : "false")
         << ",\"worker_acknowledged\":"
         << (result.inFlightReset.workerAcknowledged ? "true" : "false")
         << "},\"worker_platform\":{"
         << "\"affinity_attempted\":"
         << (result.platform.affinityAttempted ? "true" : "false")
         << ",\"affinity_error\":" << result.platform.affinityError
         << ",\"affinity_observed\":\""
         << jsonEscape(result.platform.affinityObserved)
         << "\",\"priority_attempted\":"
         << (result.platform.priorityAttempted ? "true" : "false")
         << ",\"priority_error\":" << result.platform.priorityError
         << ",\"scheduling_policy\":" << result.platform.schedulingPolicy
         << ",\"scheduling_priority\":"
         << result.platform.schedulingPriority
         << "},\"gates\":{\"correctness_pass\":"
         << (result.correctnessPass ? "true" : "false")
         << ",\"diagnostic_timing_pass\":"
         << (result.diagnosticTimingPass ? "true" : "false")
         << ",\"realtime_scheduling_observed\":"
         << (result.realtimeSchedulingObserved ? "true" : "false")
         << ",\"bounded_paced_probe_pass\":"
         << (result.correctnessPass && result.diagnosticTimingPass
                 ? "true"
                 : "false")
         << "}}";
}

int runSelfTest() {
  const std::string abc = "abc";
  Sha256 sha;
  sha.update(reinterpret_cast<const uint8_t*>(abc.data()), abc.size());
  require(sha.finish() ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 self-test failed");
  const AudioChunk first = makeAudioChunk(0);
  require(first == makeAudioChunk(0) && first != makeAudioChunk(kHopSamples),
          "deterministic audio generator self-test failed");
  const Summary summary = summarize({1.0, 2.0, 3.0});
  require(summary.mean == 2.0 && summary.maximum == 3.0,
          "summary self-test failed");
  std::cout << "cropped1024_native_async_qualifier self-test passed\n";
  return 0;
}

int runQualification(const Arguments& arguments, const char* executable) {
  require(std::filesystem::is_regular_file(arguments.model),
          "model is not a regular file");
  require(std::filesystem::file_size(arguments.model) == arguments.modelBytes,
          "model size changed");
  const std::string modelSha = sha256File(arguments.model);
  require(modelSha == arguments.modelSha256, "model SHA-256 changed");
  require(std::string(Ort::GetVersionString()) == "1.26.0",
          "ONNX Runtime must be exactly 1.26.0");
  const std::filesystem::path loadedOrt = loadedOrtPath();
  require(std::filesystem::canonical(arguments.ortLibrary) == loadedOrt,
          "loaded ONNX Runtime library differs from --ort-library");
  const std::string ortSha = sha256File(loadedOrt);
  require(ortSha == arguments.ortSha256, "ONNX Runtime library SHA-256 changed");
  require(!std::filesystem::exists(arguments.output),
          "output receipt already exists; refusing overwrite");
  require(arguments.output.parent_path().empty() ||
              std::filesystem::is_directory(arguments.output.parent_path()),
          "output receipt parent directory does not exist");

  const size_t inputCount = static_cast<size_t>(
      std::max(arguments.callbacks, arguments.warmup));
  std::vector<AudioChunk> inputs;
  inputs.reserve(inputCount);
  for (size_t index = 0; index < inputCount; ++index) {
    inputs.push_back(makeAudioChunk(
        static_cast<int64_t>(index) * static_cast<int64_t>(kHopSamples)));
  }

  std::vector<ModeResult> results;
  results.push_back(runMode(arguments, inputs));
  const MusicParityResult music = runMusicParity(arguments);

  bool allCorrectness = true;
  bool allDiagnosticTiming = true;
  bool realtimeSchedulingObserved = true;
  for (const ModeResult& result : results) {
    allCorrectness = allCorrectness && result.correctnessPass;
    allDiagnosticTiming =
        allDiagnosticTiming && result.diagnosticTimingPass;
    realtimeSchedulingObserved = realtimeSchedulingObserved &&
                                 result.realtimeSchedulingObserved;
  }
  const bool modelAndRuntimeUnchanged =
      std::filesystem::file_size(arguments.model) == arguments.modelBytes &&
      sha256File(arguments.model) == modelSha && sha256File(loadedOrt) == ortSha;
  const bool harnessPass = allCorrectness && allDiagnosticTiming &&
      music.executed && music.passed && modelAndRuntimeUnchanged;
  const std::string status =
      harnessPass ? "pass" : (!music.executed ? "incomplete_music_parity" : "fail");
  const std::filesystem::path executablePath =
      std::filesystem::canonical(executable);
  const std::string executableSha = sha256File(executablePath);
  std::ostringstream receipt;
  receipt << std::setprecision(17);
  receipt << "{\n"
          << "  \"schema_version\":1,\n"
          << "  \"kind\":\"cropped1024_linux_headless_native_async_qualification_v1\",\n"
          << "  \"status\":\"" << status
          << "\",\n"
          << "  \"scope\":{\"included\":\"Linux headless native ORT 1.26 CPU diagnostic; cropped1024 five-input five-output state ABI; complete deployed stems; paced 256-sample callback boundaries; hardened atomic queue; reset replay; partial EOF; continuous music parity\",\"excluded\":\"JUCE plugin, DAW callback or bus writer, actual host PDC behavior, packaging, Windows, macOS, and general realtime host qualification\",\"native_host_qualified\":false},\n"
          << "  \"latency_contract\":{\"sample_rate_hz\":44100,\"hop_samples\":256,\"graph_delay_samples\":256,\"async_queue_delay_samples\":256,\"graph_delay_hops\":1,\"async_queue_delay_hops\":1,\"simulated_pdc_samples\":512,\"simulated_pdc_ms\":"
          << (1000.0 * 512.0 / 44100.0) << "},\n"
          << "  \"identity\":{\"model_path\":\""
          << jsonEscape(std::filesystem::canonical(arguments.model).string())
          << "\",\"model_size\":" << arguments.modelBytes
          << ",\"model_sha256\":\"" << modelSha
          << "\",\"checkpoint_sha256\":\"" << kCheckpointSha
          << "\",\"model_state_sha256\":\"" << kModelStateSha
          << "\",\"state_family\":\"" << kFamily
          << "\",\"onnxruntime_version\":\""
          << jsonEscape(Ort::GetVersionString())
          << "\",\"onnxruntime_library_path\":\""
          << jsonEscape(loadedOrt.string())
          << "\",\"onnxruntime_library_sha256\":\"" << ortSha
          << "\",\"executable_path\":\""
          << jsonEscape(executablePath.string())
          << "\",\"executable_sha256\":\"" << executableSha
          << "\",\"model_and_runtime_unchanged_after_workload\":"
          << (modelAndRuntimeUnchanged ? "true" : "false") << "},\n"
          << "  \"runtime\":{\"platform\":\"Linux/WSL x86_64\",\"provider\":\"CPU\",\"execution_mode\":\"ORT_SEQUENTIAL\",\"graph_optimization\":\"ORT_ENABLE_ALL\",\"intra_op_threads\":"
          << arguments.threads
          << ",\"inter_op_threads\":1,\"intra_op_spinning\":false,\"inter_op_spinning\":false,\"persistent_preallocated_ort_values\":true},\n"
          << "  \"protocol\":{\"warmup_callbacks\":" << arguments.warmup
          << ",\"measured_callbacks\":" << arguments.callbacks
          << ",\"paced_callback_boundaries\":"
          << arguments.callbacks + 2
          << ",\"paced_graph_calls\":" << arguments.callbacks + 1
          << ",\"paced_graph_flush_calls\":1,\"paced_queue_only_drain_boundaries\":1"
          << ",\"deadline_ms\":" << kDeadlineMilliseconds
          << ",\"nominal_deadline_rule\":\"post-release publication timestamp must precede the next scheduled callback\",\"actual_availability_rule\":\"callback must acquire the exact due Processed slot as Reading and timestamp that observation at the actual paced boundary; availability is never reconstructed later\",\"completion_timestamp_rule\":\"capture timestamp only after the worker release-publishes Processed\",\"queue_due_rule\":\"request sequence N is due at callback boundary N+1; valid request N represents physical hop N-1\",\"reset_replay_rule\":\"first replay starts directly after an in-flight epoch reset; second replay starts after a fresh idle reset; outputs and validity must be bit exact\",\"queue_protocol\":\"hardened_atomic_slot_poll_sleep_100us\",\"priority_elevation_attempted\":false,\"realtime_scheduling_required_for_diagnostic\":false},\n"
          << "  \"results\":[";
  for (size_t index = 0; index < results.size(); ++index) {
    if (index != 0U) receipt << ',';
    writeMode(receipt, results[index]);
  }
  receipt << "],\n  \"music_parity\":";
  writeMusicParity(receipt, arguments, music);
  receipt << ",\n"
          << "  \"gate\":{\"correctness_pass\":"
          << (allCorrectness ? "true" : "false")
          << ",\"diagnostic_timing_pass\":"
          << (allDiagnosticTiming ? "true" : "false")
          << ",\"continuous_music_parity_executed\":"
          << (music.executed ? "true" : "false")
          << ",\"continuous_music_parity_pass\":"
          << (music.passed ? "true" : "false")
          << ",\"realtime_scheduling_observed\":"
          << (realtimeSchedulingObserved ? "true" : "false")
          << ",\"model_and_runtime_unchanged\":"
          << (modelAndRuntimeUnchanged ? "true" : "false")
          << ",\"bounded_linux_harness_pass\":"
          << (harnessPass ? "true" : "false")
          << ",\"native_host_qualified\":false}\n}\n";

  const std::filesystem::path temporary = arguments.output.string() + ".tmp";
  require(!std::filesystem::exists(temporary),
          "temporary output already exists; refusing overwrite");
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(output.good(), "cannot create temporary receipt");
    output << receipt.str();
    output.flush();
    require(output.good(), "failed to write receipt");
  }
  std::filesystem::rename(temporary, arguments.output);
  const std::string receiptSha = sha256File(arguments.output);
  const std::filesystem::path sidecar = arguments.output.string() + ".sha256";
  require(!std::filesystem::exists(sidecar),
          "receipt sidecar already exists; refusing overwrite");
  {
    std::ofstream output(sidecar, std::ios::binary | std::ios::trunc);
    require(output.good(), "cannot create receipt SHA sidecar");
    output << receiptSha << "  " << arguments.output.filename().string()
           << '\n';
  }
  std::cerr << "cropped1024 bounded Linux async diagnostic " << status
            << ": receipt=" << arguments.output << ", sha256=" << receiptSha
            << '\n';
  if (!music.executed) return 5;
  return harnessPass ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parseArguments(argc, argv);
    return arguments.selfTest ? runSelfTest()
                              : runQualification(arguments, argv[0]);
  } catch (const Ort::Exception& error) {
    std::cerr << "ONNX Runtime failure: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "cropped1024 native async qualifier failure: " << error.what()
              << '\n';
    return 1;
  }
}
