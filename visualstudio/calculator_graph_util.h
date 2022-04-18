#include "mediapipe/framework/calculator_graph.h"

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
                google::protobuf::TextFormat::ParseFromString(std::string(buf, file_len), &config);
      free(buf);
    }
    fclose(file);
  }
  return result;
}

#define DEFINE_LOAD_GRAPH(default_config) \
absl::Status load_calculator_graph(mediapipe::CalculatorGraph& graph, char const* config_file) { \
  if (nullptr==config_file || '\0'==config_file[0]) { \
    config_file = default_config; \
  } \
  mediapipe::CalculatorGraphConfig config; \
  if (read_config_from_pbtxt(config, config_file)) { \
    return graph.Initialize(config); \
  } \
  return absl::NotFoundError(config_file); \
}

#define DEFINE_SUBGRAPH(class_name, pbtxt) \
struct class_name : public Subgraph { \
  absl::StatusOr<CalculatorGraphConfig> GetConfig(SubgraphOptions const&) override { \
    CalculatorGraphConfig config; \
    if (read_config_from_pbtxt(config, pbtxt)) { \
      return config; \
    } \
    return absl::InternalError("Could not parse subgraph."); \
  } \
}; \
REGISTER_MEDIAPIPE_GRAPH(class_name)