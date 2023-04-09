#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/deps/file_helpers.h"

// OpenCV key input
struct UserInput {
  struct {
    int event{0}, flags{0}, x{0}, y{0};
  } mouse;
  int wait_key{-1}; // cv::waitKeyEx
  int frame_id{-1};
};

inline uint64_t get_elapsed_time_microseconds(std::chrono::system_clock::time_point const& start_time) {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now()-start_time).count();
}

inline bool read_config_from_pbtxt(mediapipe::CalculatorGraphConfig& config, char const* filename) {
  bool result = false;
  FILE* file = fopen(filename, "rb");
  if (file) {
    fseek(file, 0, SEEK_END);
    auto const file_len = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* buf = (char*) malloc(file_len);
    if (buf) {
      result = (file_len==fread(buf, 1, file_len, file)) &&
                google::protobufx::TextFormat::ParseFromString(std::string(buf, file_len), &config);
      free(buf);
    }
    fclose(file);
  }
  return result;
}

//
// https://github.com/google/mediapipe/releases/tag/v0.8.11
// We are no longer adding *.tflite model files and other large binaries to our GitHub repository.
// Instead, these models are downloaded from Google Cloud Storage. This should speed up your
// getting started experience with MediaPipe (especially if you can work of a shallow clone of the repository)
// and allows us to expand our feature set without significantly increasing the size of the repository.
//
// refer mediapipe/mediapipe/python/solutions/download_utils.py
#include <windows.h>
#pragma comment(lib, "Urlmon.lib")
inline int download_mediapipe_asset_from_GCS(char const* localpath, bool refresh=false) {
  if (mediapipe::file::Exists(localpath).ok()) {
    if (!refresh) {
      return 0;
    }
    DeleteFileA(localpath);
  }

  char url[260];
  char const* filename = strrchr(localpath, '/');
  if (!filename) {
    filename = strrchr(localpath, '\\');
  }
  sprintf(url, "https://storage.googleapis.com/mediapipe-assets/%s", filename ? (++filename):localpath);
  if (S_OK==URLDownloadToFileA(NULL, url, localpath, 0, NULL)) {
    return 1;
  }

  return -1;
}

#define DEFINE_SUBGRAPH(class_name, pbtxt) \
struct class_name : public Subgraph { \
  abslx::StatusOr<CalculatorGraphConfig> GetConfig(SubgraphOptions const&) override { \
    CalculatorGraphConfig config; \
    if (read_config_from_pbtxt(config, pbtxt)) { \
      return config; \
    } \
    return abslx::InternalError("Could not parse subgraph."); \
  } \
}; \
REGISTER_MEDIAPIPE_GRAPH(class_name)
