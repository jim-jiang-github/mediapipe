#include "../calculator_graph_util.h"

// resource root to locate tflite and other files
// see also mediapipe/mediapipe/util/resource_util_default.cc
constexpr char const* resource_root = "../";

// name of file containing text format CalculatorGraphConfig proto
constexpr char const* calculator_graph_config_file = "./objectron_desktop_cpu_live.pbtxt";

// and subgraphs...
namespace mediapipe {
DEFINE_SUBGRAPH(ObjectronCpuSubgraph, "../../mediapipe/modules/objectron/objectron_cpu.pbtxt");
  DEFINE_SUBGRAPH(ObjectDetectionOidV4Subgraph, "../../mediapipe/modules/objectron/object_detection_oid_v4_cpu.pbtxt");
  DEFINE_SUBGRAPH(BoxLandmarkSubgraph, "../../mediapipe/modules/objectron/box_landmark_cpu.pbtxt");
DEFINE_SUBGRAPH(RendererSubgraph, "../../mediapipe/graphs/object_detection_3d/subgraphs/renderer_cpu.pbtxt");
}

// detect categories
enum CATEGORY {
  CATEGORY_FOOTWEAR = 0,
  CATEGORY_CHAIR,
  CATEGORY_CAMERA,
  CATEGORY_CUP,

  CATEGORY_TOTALS
};
struct {
  char const* const box_landmark_model_path;
  char const* const allowed_labels;
  int const max_num_objects;
} categories[CATEGORY_TOTALS] = {
  { "mediapipe/modules/objectron/object_detection_3d_sneakers.tflite", "Footwear", 2 },
  { "mediapipe/modules/objectron/object_detection_3d_chair.tflite", "Chair", 4 },
  { "mediapipe/modules/objectron/object_detection_3d_camera.tflite", "Camera", 1 },
  { "mediapipe/modules/objectron/object_detection_3d_cup.tflite", "Mug", 3 },
};
CATEGORY const detect_category = CATEGORY_CUP; // free to change

abslx::Status init_calculator_graph(mediapipe::CalculatorGraph& graph) {
  mediapipe::CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, calculator_graph_config_file)) {
    auto const& cat = categories[detect_category];

    char filepath[260];
    sprintf(filepath, "../%s", cat.box_landmark_model_path);
    download_mediapipe_asset_from_GCS(filepath);
    download_mediapipe_asset_from_GCS("../mediapipe/modules/objectron/object_detection_ssd_mobilenetv2_oidv4_fp16.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/objectron/object_detection_oidv4_labelmap.txt"); // not in GCS!
    if (CATEGORY_CHAIR==detect_category) {
      download_mediapipe_asset_from_GCS("../mediapipe/modules/objectron/object_detection_3d_chair_1stage.tflite");
    } else if (CATEGORY_FOOTWEAR==detect_category) {
      download_mediapipe_asset_from_GCS("../mediapipe/modules/objectron/object_detection_3d_sneakers_1stage.tflite");
    }

    // make necessary side packets...
    std::map<std::string, mediapipe::Packet> side_packets;
    side_packets["box_landmark_model_path"] = mediapipe::MakePacket<std::string>(std::string(cat.box_landmark_model_path));
    side_packets["allowed_labels"] = mediapipe::MakePacket<std::string>(cat.allowed_labels);
    side_packets["max_num_objects"] = mediapipe::MakePacket<int>(cat.max_num_objects);

    return graph.Initialize(config, side_packets);
  }
  return abslx::NotFoundError(calculator_graph_config_file);
}

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"