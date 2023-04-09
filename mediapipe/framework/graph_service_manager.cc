#include "mediapipe/framework/graph_service_manager.h"

namespace mediapipe {

abslx::Status GraphServiceManager::SetServicePacket(
    const GraphServiceBase& service, Packet p) {
  // TODO: check service is already set?
  service_packets_[service.key] = std::move(p);
  return abslx::OkStatus();
}

Packet GraphServiceManager::GetServicePacket(
    const GraphServiceBase& service) const {
  auto it = service_packets_.find(service.key);
  if (it == service_packets_.end()) {
    return {};
  }
  return it->second;
}

}  // namespace mediapipe
