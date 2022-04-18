//
#include "mediapipe/framework/calculator_graph.h"

char const* root_graph_pbtxt = "hair_segmentation_desktop_live.pbtxt";

//
// read graph config from .pbtxt
bool read_config_from_pbtxt(mediapipe::CalculatorGraphConfig& config, char const* filename) {
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

namespace mediapipe {

// to load calculator graph
absl::Status load_calculator_graph(mediapipe::CalculatorGraph& graph) {
  CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, root_graph_pbtxt)) {
    return graph.Initialize(config);
  }

  return NotFoundError(root_graph_pbtxt);
}

}  // namespace mediapipe
