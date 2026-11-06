// SPDX-License-Identifier: GPL-3.0-or-later

// Shared mock-params texture cache context for the renderer-level regression tests
// (clear_ordering_harness.cpp, tic_binding_limits.cpp). A real VideoCommon::TextureCache<P>
// over real guest memory and a real channel, with the renderer side recorded: the copy/write
// seam is a recording MockRuntime and the Scheduler's deferred-clear state machine a contract
// model (Vulkan::Scheduler cannot be constructed without a device, see the harness notes in
// /home/vouion/oot-war-room/drafts/clear-ordering-harness.md).

#pragma once

#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/alignment.h"
#include "common/page_table.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {
/// Drives the private copy paths of TextureCache<P> from the harness below.
struct TextureCacheTestPeer {
    template <class P>
    static ImageId InsertImage(TextureCache<P>& cache, const ImageInfo& info, GPUVAddr gpu_addr) {
        return cache.InsertImage(info, gpu_addr, RelaxedOptions{});
    }

    template <class P>
    static void CopyImage(TextureCache<P>& cache, ImageId dst_id, ImageId src_id,
                          std::vector<ImageCopy> copies) {
        cache.CopyImage(dst_id, src_id, std::move(copies));
    }

    template <class P>
    static std::pair<u32, u32> PrepareDmaImage(TextureCache<P>& cache, ImageId dst_id,
                                                GPUVAddr base_addr, bool mark_as_modified) {
        return cache.PrepareDmaImage(dst_id, base_addr, mark_as_modified);
    }

    template <class P>
    static typename TextureCache<P>::Image& Image(TextureCache<P>& cache, ImageId id) {
        return cache.slot_images[id];
    }
};
} // namespace VideoCommon

namespace TextureCacheTests {

using VideoCommon::BufferImageCopy;
using VideoCommon::ImageCopy;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageId;
using VideoCommon::ImageInfo;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewInfo;
using VideoCommon::ImageViewType;
using VideoCommon::NullImageParams;
using VideoCommon::NullImageViewParams;
using VideoCommon::RelaxedOptions;
using VideoCommon::RenderTargets;
using VideoCommon::SubresourceBase;
using VideoCommon::SubresourceLayers;
using VideoCommon::TextureCacheTestPeer;
using VideoCore::Surface::PixelFormat;

/// Ordering-relevant events, in the order the seam produced them.
enum class Event {
    DeferColorClear,       ///< the rasterizer parks a full colour clear
    DeferDepthStencilClear, ///< the rasterizer parks a full depth/stencil clear
    FlushRequest,          ///< runtime.FlushDeferredClear() (a texture-cache call site)
    RealizeStandalone,     ///< the parked clear lands as its own clear-only pass
    RealizeLoadOp,         ///< the parked clear lands as the next draw's load-op
    BeginPass,
    EndPass,
    BlitCopy,  ///< runtime.BlitImage: the actual 2D copy
    Copy,      ///< runtime.CopyImage: the actual alias copy
    Sample,    ///< fixture marker: a draw samples the copied image
};

inline const char* EventName(Event event) {
    switch (event) {
    case Event::DeferColorClear:
        return "DeferColorClear";
    case Event::DeferDepthStencilClear:
        return "DeferDepthStencilClear";
    case Event::FlushRequest:
        return "FlushRequest";
    case Event::RealizeStandalone:
        return "RealizeStandalone";
    case Event::RealizeLoadOp:
        return "RealizeLoadOp";
    case Event::BeginPass:
        return "BeginPass";
    case Event::EndPass:
        return "EndPass";
    case Event::BlitCopy:
        return "BlitCopy";
    case Event::Copy:
        return "Copy";
    case Event::Sample:
        return "Sample";
    }
    return "?";
}

/// One-line rendering of an event stream for Catch2 diagnostics.
inline std::string EventNames(const std::vector<Event>& events) {
    std::string out;
    for (const Event event : events) {
        if (!out.empty()) {
            out += " -> ";
        }
        out += EventName(event);
    }
    return out;
}

struct EventLog {
    std::vector<Event> events;

    void Record(Event event) {
        events.push_back(event);
    }

    [[nodiscard]] std::optional<size_t> FirstIndex(Event event) const {
        for (size_t i = 0; i < events.size(); ++i) {
            if (events[i] == event) {
                return i;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<size_t> LastIndex(Event event) const {
        for (size_t i = events.size(); i > 0; --i) {
            if (events[i - 1] == event) {
                return i - 1;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] size_t Count(Event event) const {
        size_t count = 0;
        for (const Event recorded : events) {
            count += recorded == event;
        }
        return count;
    }

    void Clear() {
        events.clear();
    }
};

/// Recording stand-in for Vulkan::Scheduler's deferred-clear state machine
/// (DeferColorClear/DeferDepthStencilClear/FlushDeferredClear/RequestRenderpass and the realize
/// at the top of EndRenderPass, vk_scheduler.cpp:347-432 and :858). A parked clear is fused as
/// the framebuffer's next render pass (load-op), forced into a standalone clear-only pass by
/// FlushDeferredClear, and realized before EndRenderPass drops a pending pass.
class DeferredClearModel {
public:
    explicit DeferredClearModel(EventLog& log_) : log{log_} {}

    void DeferColorClear(const void* framebuffer) {
        log.Record(Event::DeferColorClear);
        Park(framebuffer);
    }

    void DeferDepthStencilClear(const void* framebuffer) {
        log.Record(Event::DeferDepthStencilClear);
        Park(framebuffer);
    }

    /// TextureCacheRuntime::FlushDeferredClear: realize, then end the pass.
    void FlushDeferredClear() {
        log.Record(Event::FlushRequest);
        if (parked) {
            Realize();
            EndPass();
        }
    }

    /// A draw/load of `framebuffer` consumes a clear parked on it as its load-op.
    void RequestRenderpass(const void* framebuffer) {
        if (parked && parked_framebuffer == framebuffer) {
            log.Record(Event::RealizeLoadOp);
            parked = false;
            active = true;
            log.Record(Event::BeginPass);
            return;
        }
        EndPass();
        log.Record(Event::BeginPass);
        active = true;
    }

    [[nodiscard]] bool Pending() const {
        return parked;
    }

private:
    void Park(const void* framebuffer) {
        // A clear on another framebuffer realizes the parked one first (Defer*Clear).
        if (parked && parked_framebuffer != framebuffer) {
            Realize();
            EndPass();
        }
        parked_framebuffer = framebuffer;
        parked = true;
    }

    void Realize() {
        log.Record(Event::RealizeStandalone);
        parked = false;
    }

    void EndPass() {
        // EndRenderPass realizes a parked clear before it can be lost.
        if (parked) {
            Realize();
        }
        if (active) {
            log.Record(Event::EndPass);
            active = false;
        }
    }

    EventLog& log;
    const void* parked_framebuffer{};
    bool parked{};
    bool active{};
};

/// Staging buffer handle, standing in for Vulkan::StagingBufferRef.
struct MockStaging {
    u32 buffer{};
    u64 device_address{};
    u64 offset{};
    std::span<u8> mapped_span;
    u32 log2_level{};
    u64 index{};
};

class MockRuntime;

struct MockImage : VideoCommon::ImageBase {
    explicit MockImage(MockRuntime&, const ImageInfo& info, GPUVAddr gpu_addr_, DAddr cpu_addr_)
        : ImageBase(info, gpu_addr_, cpu_addr_) {}
    explicit MockImage(const NullImageParams& params) : ImageBase(params) {}

    MockImage(const MockImage&) = delete;
    MockImage& operator=(const MockImage&) = delete;
    MockImage(MockImage&&) = default;
    MockImage& operator=(MockImage&&) = default;

    void UploadMemory(const MockStaging&, std::span<const BufferImageCopy>) {}
    void DownloadMemory(const MockStaging&, std::span<const BufferImageCopy>) {}
    void DownloadMemory(u32, size_t, std::span<const BufferImageCopy>) {}
    void DownloadMemory(std::span<const u32>, std::span<const size_t>,
                        std::span<const BufferImageCopy>) {}

    bool ScaleUp(bool ignore = false) {
        return ignore;
    }
    bool ScaleDown(bool ignore = false) {
        return ignore;
    }

    u64 allocation_tick{};
};

struct MockImageView : VideoCommon::ImageViewBase {
    explicit MockImageView(MockRuntime&, const ImageViewInfo& info, ImageId image_id_,
                           MockImage& image, const Common::SlotVector<MockImage>&)
        : ImageViewBase(info, image.info, image_id_, image.gpu_addr) {}
    explicit MockImageView(MockRuntime&, const ImageInfo& info, const ImageViewInfo& view_info,
                           GPUVAddr addr)
        : ImageViewBase(info, view_info, addr) {}
    explicit MockImageView(MockRuntime&, const NullImageViewParams& params)
        : ImageViewBase(params) {}

    MockImageView(const MockImageView&) = delete;
    MockImageView& operator=(const MockImageView&) = delete;
    MockImageView(MockImageView&&) = default;
    MockImageView& operator=(MockImageView&&) = default;
};

struct MockSampler {
    MockSampler(MockRuntime&, const Tegra::Texture::TSCEntry&) {}

    MockSampler(const MockSampler&) = delete;
    MockSampler& operator=(const MockSampler&) = delete;
    MockSampler(MockSampler&&) = default;
    MockSampler& operator=(MockSampler&&) = default;
};

struct MockFramebuffer {
    MockFramebuffer(MockRuntime&, std::span<MockImageView*, VideoCommon::NUM_RT> color_buffers,
                    MockImageView* depth_buffer, const RenderTargets& key_)
        : key{key_}, has_depth{depth_buffer != nullptr} {}

    MockFramebuffer(const MockFramebuffer&) = delete;
    MockFramebuffer& operator=(const MockFramebuffer&) = delete;
    MockFramebuffer(MockFramebuffer&&) = default;
    MockFramebuffer& operator=(MockFramebuffer&&) = default;

    RenderTargets key;
    bool has_depth{};
};

struct MockImageAlloc : VideoCommon::ImageAllocBase {};

/// Records the copy/write events and forwards the deferred-clear seam to the model.
class MockRuntime {
public:
    MockRuntime(EventLog& log_, DeferredClearModel& clears_) : log{log_}, clears{clears_} {}

    void Finish() {}
    void TickFrame() {}

    MockStaging UploadStagingBuffer(size_t size, bool deferred = false) {
        return Staging(size);
    }
    MockStaging DownloadStagingBuffer(size_t size, bool deferred = false) {
        return Staging(size);
    }
    void FreeDeferredStagingBuffer(MockStaging&) {}

    void FlushDeferredClear() {
        clears.FlushDeferredClear();
    }

    u64 GetDeviceLocalMemory() const {
        return 4ULL << 30;
    }
    u64 GetDeviceMemoryUsage() const {
        return 0;
    }
    u64 GetDeviceMemoryBudget() const {
        return 4ULL << 30;
    }
    u64 GetDeviceMemorySlack() const {
        return 1ULL << 30;
    }
    bool CanReportMemoryUsage() const {
        return true;
    }
    u32 MemoryPressureEvents() const {
        return 0;
    }
    void LogMemoryStatistics() const {}
    u64 CurrentTick() const {
        return 0;
    }
    bool IsFree(u64) const {
        return true;
    }

    // MSAA uploads are skipped (RefreshContents stops before the upload math), which keeps the
    // MSAA fixture of the first scenario out of the swizzle code.
    bool CanUploadMSAA() const noexcept {
        return false;
    }
    bool CanDownloadMsaa(const ImageInfo&) const {
        return false;
    }

    void BlitImage(MockFramebuffer*, MockImageView&, MockImageView&,
                   const VideoCommon::Region2D&, const VideoCommon::Region2D&,
                   Tegra::Engines::Fermi2D::Filter, Tegra::Engines::Fermi2D::Operation) {
        log.Record(Event::BlitCopy);
    }
    void BlitFramebuffer(MockFramebuffer*, MockFramebuffer*, const VideoCommon::Region2D&,
                         const VideoCommon::Region2D&, Tegra::Engines::Fermi2D::Filter,
                         Tegra::Engines::Fermi2D::Operation) {
        log.Record(Event::BlitCopy);
    }
    void CopyImage(MockImage&, MockImage&, std::span<const ImageCopy>) {
        log.Record(Event::Copy);
    }
    void CopyImageMSAA(MockImage&, MockImage&, std::span<const ImageCopy>) {
        log.Record(Event::Copy);
    }
    bool ShouldReinterpret(MockImage&, MockImage&) {
        return false;
    }
    void ReinterpretImage(MockImage&, MockImage&, std::span<const ImageCopy>) {
        log.Record(Event::Copy);
    }
    void ConvertImage(MockFramebuffer*, MockImageView&, MockImageView&) {
        log.Record(Event::Copy);
    }

    void InsertUploadMemoryBarrier() {}
    void TransitionImageLayout(MockImage&) {}
    void BarrierFeedbackLoop() {}
    void AccelerateImageUpload(MockImage&, const MockStaging&,
                               std::span<const VideoCommon::SwizzleParameters>, u32, u32) {}
    bool HasBrokenTextureViewFormats() const noexcept {
        return false;
    }
    bool HasNativeBgr() const noexcept {
        return true;
    }

private:
    MockStaging Staging(size_t size) {
        staging_store.emplace_back(size);
        return MockStaging{.mapped_span = std::span<u8>{staging_store.back()}};
    }

    EventLog& log;
    DeferredClearModel& clears;
    // Stable storage for the spans handed out as staging buffers.
    std::deque<std::vector<u8>> staging_store;
};

struct MockParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;
    static constexpr bool HAS_EMULATED_COPIES = false;
    static constexpr bool HAS_DEVICE_MEMORY_INFO = true;
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = false;
    static constexpr bool HAS_MSAA_DOWNLOADS = false;

    using Runtime = MockRuntime;
    using Image = MockImage;
    using ImageAlloc = MockImageAlloc;
    using ImageView = MockImageView;
    using Sampler = MockSampler;
    using Framebuffer = MockFramebuffer;
    using AsyncBuffer = MockStaging;
    using BufferType = u32;
};

using MockTextureCache = VideoCommon::TextureCache<MockParams>;

/// Texture cache over real guest memory and a real channel, with the renderer side recorded.
struct Harness {
    /// The memory manager calls into the rasterizer on mapping changes.
    class NullRasterizer final : public VideoCore::RasterizerInterface {
    public:
        void Draw(bool, u32) override {}
        void DrawTexture() override {}
        void Clear(u32) override {}
        void DispatchCompute() override {}
        void ResetCounter(VideoCommon::QueryType) override {}
        void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
                   u32) override {}
        void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
        void DisableGraphicsUniformBuffer(size_t, u32) override {}
        void SignalFence(std::function<void()>&&) override {}
        void SyncOperation(std::function<void()>&&) override {}
        void SignalSyncPoint(u32) override {}
        void SignalReference() override {}
        void ReleaseFences(bool) override {}
        void FlushAll() override {}
        void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
        bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override {
            return false;
        }
        VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
            return {addr, addr + size, false};
        }
        void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
        void OnCacheInvalidation(DAddr, u64) override {}
        bool OnCPUWrite(DAddr, u64) override {
            return false;
        }
        void InvalidateGPUCache() override {}
        void UnmapMemory(DAddr, u64) override {}
        void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
        void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
        void WaitForIdle() override {}
        void FragmentBarrier() override {}
        void TiledCacheBarrier() override {}
        void FlushCommands() override {}
        void TickFrame() override {}
        Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override {
            std::terminate();
        }
        void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}
    };

    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    NullRasterizer rasterizer;
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Control::ChannelState channel{0};
    EventLog log;
    DeferredClearModel scheduler{log};
    MockRuntime runtime{log, scheduler};
    MockTextureCache cache{runtime, device_memory};
    // Guest memory that data actually round-trips through (the process-backed recipe of
    // maxwell_fast_method.cpp): arena pages behind a guest process mapping behind the device
    // mapping behind the GPU mapping.
    std::optional<Kernel::KProcess> process;
    Core::Asid asid{};

    Harness() {
        system.Initialize();
        process.emplace(system.Kernel());
        // One page table for all mappings; per-mapping entries are added in MapGuest. Size it
        // before SetCurrentPageTable, which binds the table's arrays.
        process->GetPageTable().GetImpl().Resize(32, 12);
        process->GetMemory().SetCurrentPageTable(*process);
        asid = device_memory.RegisterProcess(&process->GetMemory());
        device_memory.BindInterface(&rasterizer);
        memory.BindRasterizer(&rasterizer);
        channel.memory_manager = &memory;
        channel.Init(system, 0);
        cache.CreateChannel(channel);
        cache.BindToChannel(channel.bind_id);
    }

    ~Harness() {
        device_memory.UnregisterProcess(asid);
        system.Kernel().UnregisterKernelObject(&*process);
    }

    Harness(const Harness&) = delete;
    Harness& operator=(const Harness&) = delete;

    /// Maps `size` bytes of guest memory at `gpu_addr`, backed by arena pages.
    void MapGuest(GPUVAddr gpu_addr, size_t size) {
        const DAddr dev_addr = next_dev_addr;
        const VAddr guest_addr = next_guest_addr;
        const size_t aligned = Common::AlignUp(size, GuestAlign);
        auto& table = process->GetPageTable().GetImpl();
        auto* const backing = physical.GetPointerFromRaw<u8>(next_arena_offset);
        for (size_t offset = 0; offset < aligned; offset += 0x1000) {
            table.entries.GetAndFault((guest_addr + offset) >> 12)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(backing) - guest_addr);
        }
        device_memory.Map(dev_addr, guest_addr, aligned, asid);
        memory.Map(gpu_addr & ~(GuestAlign - 1), dev_addr, aligned, PTEKind, true);
        next_dev_addr += aligned;
        next_guest_addr += aligned;
        next_arena_offset += aligned;
    }

    /// Maps guest memory for an image and inserts it through the cache's own creation path.
    ImageId Insert(const ImageInfo& info, GPUVAddr gpu_addr) {
        const size_t size = Common::AlignUp(VideoCommon::CalculateGuestSizeInBytes(info), GuestAlign);
        MapGuest(gpu_addr, size);
        return TextureCacheTestPeer::InsertImage(cache, info, gpu_addr);
    }

    static constexpr u64 GuestAlign = 0x10000;
    static constexpr Tegra::PTEKind PTEKind = Tegra::PTEKind::PITCH;
    DAddr next_dev_addr = 0x1'0000;
    VAddr next_guest_addr = 0x10'0000;
    u64 next_arena_offset = 0x10'0000;
};


} // namespace TextureCacheTests
