#pragma once

#include "stm_snapshot_data.hpp"
#include "../memory/stm.hpp"

#include <string>
#include <cstdint>

namespace brain19 {

// Binary format constants
constexpr uint32_t STM_SNAPSHOT_MAGIC   = 0x53544D53; // "STMS"
constexpr uint16_t STM_SNAPSHOT_VERSION = 1;

class STMSnapshotManager {
public:
    explicit STMSnapshotManager(size_t max_snapshots = 5);

    bool create_snapshot(ShortTermMemory& stm, const std::string& path);
    bool load_snapshot(const std::string& path, STMSnapshotData& out);
    void apply_snapshot(ShortTermMemory& stm, const STMSnapshotData& data);
    void rotate_snapshots(const std::string& directory, const std::string& prefix);

private:
    size_t max_snapshots_;
};

} // namespace brain19
