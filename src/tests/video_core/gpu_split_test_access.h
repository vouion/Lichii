// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

#include "common/common_types.h"
#include "video_core/gpu_split.h"

namespace Core {
class System;
}

namespace VideoCommon {

/// GpuSplit's friend: drives its ring from tests without a back thread.
struct GpuSplitTestAccess {
    static void Initialize(GpuSplit& split, Core::System& system, u64 position = 0) {
        split.system = &system;
        split.ring = std::make_unique<u64[]>(GpuSplit::RingWords);
        split.write_position = position;
        split.consumed_cache = position;
        split.published_local = position;
        split.publish_every = GpuSplit::RingWords;
    }

    /// As Initialize: the note positions restart from `position` too.
    static void ResetNotes(GpuSplit& split, u64 position = 0) {
        split.last_writer_position = position;
        split.last_memory_writer_position = position;
    }

    static u64 End(const GpuSplit& split) {
        return split.write_position;
    }

    static u64 NearWrap() {
        return GpuSplit::RingWords - 1;
    }

    static u64 Execute(GpuSplit& split, u64 position, u64 available) {
        const size_t index = position & GpuSplit::RingMask;
        const size_t size = split.Execute(
            &split.ring[index], std::min<u64>(available - position, GpuSplit::RingWords - index));
        return position + (size != 0 ? size : GpuSplit::RingWords - index);
    }
};

} // namespace VideoCommon
