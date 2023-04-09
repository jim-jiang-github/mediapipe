#ifndef MEDIAPIPE_FRAMEWORK_TOOL_GRAPH_PROCESSOR_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_GRAPH_PROCESSOR_H_

#include <memory>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/switch/packet_processor.h"

namespace mediapipe {

// Processes MediaPipe Packets using a MediaPipe CalculatorGraph.
class GraphProcessor : public PacketProcessor {
 public:
  GraphProcessor() = default;

  // Configures this GraphProcessor to create a run a CalculatorGraph.
  abslx::Status Initialize(CalculatorGraphConfig graph_config);

 public:
  // The PacketProcessor interface.
  abslx::Status AddPacket(CollectionItemId id, Packet packet) override;
  std::shared_ptr<tool::TagMap> InputTags() override;
  abslx::Status SetSidePacket(CollectionItemId id, Packet packet) override;
  std::shared_ptr<tool::TagMap> SideInputTags() override;
  void SetConsumer(PacketConsumer* consumer) override;
  void SetSideConsumer(SidePacketConsumer* consumer) override;
  abslx::Status Start() override;
  abslx::Status Shutdown() override;
  abslx::Status WaitUntilIdle() override;

 private:
  // Sends a tagged output packet.
  abslx::Status SendPacket(CollectionItemId id, Packet packet);

  // Observes output packets from the calculator graph.
  abslx::Status ObserveGraph() ABSL_SHARED_LOCKS_REQUIRED(graph_mutex_);

  // Blocks until this GraphProcessor is initialized.
  abslx::Status WaitUntilInitialized();

 private:
  CalculatorGraphConfig graph_config_;
  std::shared_ptr<tool::TagMap> graph_input_map_;
  std::shared_ptr<tool::TagMap> graph_output_map_;
  std::map<CollectionItemId, CollectionItemId> consumer_ids_;

  PacketConsumer* consumer_ = nullptr;
  std::map<std::string, Packet> side_packets_;
  std::unique_ptr<CalculatorGraph> graph_ ABSL_GUARDED_BY(graph_mutex_) =
      nullptr;
  abslx::Mutex graph_mutex_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_GRAPH_PROCESSOR_H_
