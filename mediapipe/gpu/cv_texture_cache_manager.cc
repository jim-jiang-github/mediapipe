// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/gpu/cv_texture_cache_manager.h"

#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

void CvTextureCacheManager::FlushTextureCaches() {
  abslx::MutexLock lock(&mutex_);
  for (const auto& cache : texture_caches_) {
#if TARGET_OS_OSX
    CVOpenGLTextureCacheFlush(*cache, 0);
#else
    CVOpenGLESTextureCacheFlush(*cache, 0);
#endif  // TARGET_OS_OSX
  }
}

void CvTextureCacheManager::RegisterTextureCache(CVTextureCacheType cache) {
  abslx::MutexLock lock(&mutex_);

  CHECK(std::find(texture_caches_.begin(), texture_caches_.end(), cache) ==
        texture_caches_.end())
      << "Attempting to register a texture cache twice";
  texture_caches_.emplace_back(cache);
}

void CvTextureCacheManager::UnregisterTextureCache(CVTextureCacheType cache) {
  abslx::MutexLock lock(&mutex_);

  auto it = std::find(texture_caches_.begin(), texture_caches_.end(), cache);
  CHECK(it != texture_caches_.end())
      << "Attempting to unregister an unknown texture cache";
  texture_caches_.erase(it);
}

CvTextureCacheManager::~CvTextureCacheManager() {
  CHECK_EQ(texture_caches_.size(), 0)
      << "Failed to unregister texture caches before deleting manager";
}

}  // namespace mediapipe
