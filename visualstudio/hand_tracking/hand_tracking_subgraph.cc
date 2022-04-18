#include "mediapipe/framework/subgraph.h"
#include "mediapipe/framework/calculator_graph.h"

char const* root_graph_pbtxt = "hand_tracking_desktop_live.pbtxt";

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

#define DEFINE_SUBGRAPH(class_name, pbtxt)                                           \
struct class_name : public Subgraph {                                                \
  absl::StatusOr<CalculatorGraphConfig> GetConfig(SubgraphOptions const&) override { \
    CalculatorGraphConfig config;                                                    \
    if (read_config_from_pbtxt(config, pbtxt)) {                                     \
      return config;                                                                 \
    }                                                                                \
    return absl::InternalError("Could not parse subgraph.");                         \
  }                                                                                  \
};                                                                                   \
REGISTER_MEDIAPIPE_GRAPH(class_name)

//
// root graph: hand_tracking_desktop_live.pbtxt
DEFINE_SUBGRAPH(HandLandmarkTrackingCpu, "hand_landmark_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionCpu, "palm_detection_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionModelLoader, "palm_detection_model_loader.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionDetectionToRoi, "palm_detection_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarkCpu, "hand_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkModelLoader, "hand_landmark_model_loader.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarkLandmarksToRoi, "hand_landmark_landmarks_to_roi.pbtxt");

DEFINE_SUBGRAPH(HandRendererSubgraph, "hand_renderer_cpu.pbtxt");

// to load calculator graph
absl::Status load_calculator_graph(mediapipe::CalculatorGraph& graph) {
  CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, root_graph_pbtxt)) {
    return graph.Initialize(config);
  }

  return NotFoundError(root_graph_pbtxt);
}

}  // namespace mediapipe
