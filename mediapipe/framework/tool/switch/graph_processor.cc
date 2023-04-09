#include "mediapipe/framework/tool/switch/graph_processor.h"

#include "absl/synchronization/mutex.h"

namespace mediapipe {

// TODO: add support for input and output side packets.
abslx::Status GraphProcessor::Initialize(CalculatorGraphConfig graph_config) {
  graph_config_ = graph_config;

  ASSIGN_OR_RETURN(graph_input_map_,
                   tool::TagMap::Create(graph_config_.input_stream()));
  ASSIGN_OR_RETURN(graph_output_map_,
                   tool::TagMap::Create(graph_config_.output_stream()));
  return abslx::OkStatus();
}

abslx::Status GraphProcessor::AddPacket(CollectionItemId id, Packet packet) {
  abslx::MutexLock lock(&graph_mutex_);
  const std::string& stream_name = graph_input_map_->Names().at(id.value());
  return graph_->AddPacketToInputStream(stream_name, packet);
}

std::shared_ptr<tool::TagMap> GraphProcessor::InputTags() {
  return graph_input_map_;
}

abslx::Status GraphProcessor::SendPacket(CollectionItemId id, Packet packet) {
  MP_RETURN_IF_ERROR(WaitUntilInitialized());
  auto it = consumer_ids_.find(id);
  if (it == consumer_ids_.end()) {
    return abslx::NotFoundError(
        abslx::StrCat("Consumer stream not found: ", id.value()));
  }
  return consumer_->AddPacket(it->second, packet);
}

void GraphProcessor::SetConsumer(PacketConsumer* consumer) {
  abslx::MutexLock lock(&graph_mutex_);
  consumer_ = consumer;
  auto input_map = consumer_->InputTags();
  for (auto id = input_map->BeginId(); id != input_map->EndId(); ++id) {
    auto tag_index = input_map->TagAndIndexFromId(id);
    auto stream_id = graph_input_map_->GetId(tag_index.first, tag_index.second);
    consumer_ids_[stream_id] = id;
  }
}

abslx::Status GraphProcessor::ObserveGraph() {
  for (auto id = graph_output_map_->BeginId(); id != graph_output_map_->EndId();
       ++id) {
    std::string stream_name = graph_output_map_->Names().at(id.value());
    MP_RETURN_IF_ERROR(graph_->ObserveOutputStream(
        stream_name,
        [this, id](const Packet& packet) { return SendPacket(id, packet); },
        true));
  }
  return abslx::OkStatus();
}

abslx::Status GraphProcessor::WaitUntilInitialized() {
  abslx::MutexLock lock(&graph_mutex_);
  auto is_initialized = [this]() ABSL_SHARED_LOCKS_REQUIRED(graph_mutex_) {
    return graph_ != nullptr && consumer_ != nullptr;
  };
  graph_mutex_.AwaitWithTimeout(abslx::Condition(&is_initialized),
                                abslx::Seconds(4));
  RET_CHECK(is_initialized()) << "GraphProcessor initialization timed out.";
  return abslx::OkStatus();
}

abslx::Status GraphProcessor::Start() {
  abslx::MutexLock lock(&graph_mutex_);
  graph_ = std::make_unique<CalculatorGraph>();

  // The graph is validated here with its specified inputs and output.
  MP_RETURN_IF_ERROR(graph_->Initialize(graph_config_, side_packets_));
  MP_RETURN_IF_ERROR(ObserveGraph());
  MP_RETURN_IF_ERROR(graph_->StartRun({}));
  return abslx::OkStatus();
}

abslx::Status GraphProcessor::Shutdown() {
  abslx::MutexLock lock(&graph_mutex_);
  if (!graph_) {
    return abslx::OkStatus();
  }
  MP_RETURN_IF_ERROR(graph_->CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph_->WaitUntilDone());
  graph_ = nullptr;
  return abslx::OkStatus();
}

abslx::Status GraphProcessor::WaitUntilIdle() {
  abslx::MutexLock lock(&graph_mutex_);
  return graph_->WaitUntilIdle();
}

// TODO
abslx::Status GraphProcessor::SetSidePacket(CollectionItemId id, Packet packet) {
  return abslx::OkStatus();
}
// TODO
std::shared_ptr<tool::TagMap> GraphProcessor::SideInputTags() {
  return nullptr;
}
// TODO
void GraphProcessor::SetSideConsumer(SidePacketConsumer* consumer) {}

}  // namespace mediapipe
