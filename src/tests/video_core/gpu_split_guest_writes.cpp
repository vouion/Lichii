// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The guest-write watermark (GpuSplit::GuestWritesConsumed) behind the macro DMA read's no-wait
// path (Maxwell3D::FrontDmaRead): a read may skip the round trip only after every dispatched
// entry that can write guest memory has run on the back. These tests pin the note coverage that
// makes that sound - writer work, memory-writer work, and the draws a draw-manager method can
// issue (they write through shader stores and stream out).

#include <atomic>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "tests/core/system_test_context.h"
#include "tests/video_core/gpu_split_test_access.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace {

using Tegra::Engines::Maxwell3D;

constexpr u32 DrawEnd = MAXWELL3D_REG_INDEX(draw.end);
constexpr u32 DrawBegin = MAXWELL3D_REG_INDEX(draw.begin);

// Starts the real split (its back thread) and restores the singleton after the test.
struct SplitRun {
    VideoCommon::GpuSplit& split{VideoCommon::GpuSplit::Get()};
    Settings::RendererBackend backend{Settings::values.renderer_backend.GetValue()};
    Settings::DmaAccuracy dma{Settings::values.dma_accuracy.GetValue()};
    Settings::GpuAccuracy accuracy{Settings::values.current_gpu_accuracy};
    bool native{Tegra::g_nx2_ampere.exchange(true)};

    SplitRun(Core::System& system) {
        Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
        Settings::values.dma_accuracy.SetValue(Settings::DmaAccuracy::Unsafe);
        Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Low;
        split.MaybeStart(system);
    }
    ~SplitRun() {
        split.Stop();
        Tegra::Nx2ClearGpuWrittenPages();
        Tegra::g_nx2_ampere = native;
        Settings::values.renderer_backend.SetValue(backend);
        Settings::values.dma_accuracy.SetValue(dma);
        Settings::values.current_gpu_accuracy = accuracy;
    }
};

} // Anonymous namespace

TEST_CASE("Guest-write watermark waits for queued memory-writing work",
          "[video_core][gpu_split][guest_writes]") {
    ScopedSystemTestContext context;
    Core::System system;
    SplitRun run{system};
    if (!run.split.Active()) {
        SKIP("GPU split is disabled by EDEN_NX2_SPLIT; use 1 or 2 to exercise the back thread");
    }
    auto& split = run.split;
    // The work blocks until the gate opens, so the back cannot run past its note meanwhile.
    std::atomic<bool> gate{false};
    std::atomic<bool> ran{false};
    split.Defer([&gate, &ran] {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ran.store(true, std::memory_order_release);
    });
    split.NoteMemoryWriter();
    CHECK_FALSE(split.GuestWritesConsumed());
    gate.store(true, std::memory_order_release);
    (void)split.WaitForMemoryWriters();
    CHECK(ran.load(std::memory_order_acquire));
    CHECK(split.GuestWritesConsumed());

    // The same for compute and copy dispatches (the writer note).
    std::atomic<bool> gate2{false};
    split.Defer([&gate2] {
        while (!gate2.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    });
    split.NoteWriter();
    CHECK_FALSE(split.GuestWritesConsumed());
    gate2.store(true, std::memory_order_release);
    (void)split.WaitForWriters();
    CHECK(split.GuestWritesConsumed());
}

TEST_CASE("Draw-manager methods on the split front note guest writes",
          "[video_core][gpu_split][guest_writes]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Maxwell3D maxwell{memory};
    maxwell.BindRasterizer(&rasterizer);
    VideoCommon::Dirty::SetupDirtyFlags(maxwell.dirty.tables);
    system.Initialize();

    VideoCommon::GpuSplit split;
    Access::Initialize(split, system);
    Access::ResetNotes(split);
    CHECK(split.GuestWritesConsumed());

    // A method the draw manager does not handle itself notes nothing.
    maxwell.FrontCallMethod(system, split, MAXWELL3D_REG_INDEX(falcon[4]), 1, true);
    CHECK(split.GuestWritesConsumed());

    // DrawEnd draws (or defers an instance draw): its issue can write guest memory, so the
    // watermark must cover the entry the front sent for it.
    maxwell.FrontCallMethod(system, split, DrawEnd, 0, true);
    CHECK_FALSE(split.GuestWritesConsumed());

    // DrawBegin flushes the deferred draw; batched draws note too.
    Access::ResetNotes(split);
    maxwell.FrontCallMethod(system, split, DrawBegin, 0, true);
    CHECK_FALSE(split.GuestWritesConsumed());
    Access::ResetNotes(split);
    const u32 words[]{0, 0};
    maxwell.FrontCallMultiMethod(system, split, DrawEnd, words, 2, 2);
    CHECK_FALSE(split.GuestWritesConsumed());
}
