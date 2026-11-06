// SPDX-License-Identifier: GPL-3.0-or-later

// Renderer-level regression harness for deferred clear vs texture-cache copy ordering,
// requested on PR #185 (zortos293/Lichii). A parked deferred clear becomes its framebuffer's
// next render pass, so a non-pass image write (2D blit, image copy, DMA upload) that lands
// between the clear and that pass must realize the clear first: realizing afterwards wipes what
// the copy wrote (issue #101, the Ocarina of Time terrain layer). Rejected copies, on the other
// hand, write nothing and must leave the clear parked for the next draw's load-op.
//
// The texture cache copy paths run for real against a mock runtime; the Scheduler's deferred
// clear state machine is recorded through a model of its contract, since Vulkan::Scheduler
// cannot be constructed without a Vulkan device (see the harness notes in
// /home/vouion/oot-war-room/drafts/clear-ordering-harness.md).

#include <cstdint>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/fermi_2d.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/util.h"

#include "tests/video_core/texture_cache_test_context.h"

namespace {

using namespace TextureCacheTests;
using VideoCommon::ImageInfo;
using VideoCommon::RelaxedOptions;
using VideoCommon::SubresourceLayers;
using VideoCommon::TextureCacheTestPeer;
using VideoCore::Surface::PixelFormat;

using Surface = Tegra::Engines::Fermi2D::Surface;
using BlitConfig = Tegra::Engines::Fermi2D::Config;

/// A block-linear 2D surface, like the ones the Fermi2D engine passes to BlitImage.
Surface MakeSurface(GPUVAddr addr, u32 width, u32 height,
                    Tegra::RenderTargetFormat format = Tegra::RenderTargetFormat::R32_FLOAT) {
    Surface surface{};
    surface.format = format;
    surface.linear = Tegra::Engines::Fermi2D::MemoryLayout::BlockLinear;
    surface.block_width.Assign(0);
    surface.block_height.Assign(4);
    surface.block_depth.Assign(0);
    surface.depth = 1;
    surface.layer = 0;
    surface.pitch = 0;
    surface.width = width;
    surface.height = height;
    surface.addr_upper = static_cast<u32>(addr >> 32);
    surface.addr_lower = static_cast<u32>(addr);
    return surface;
}

ImageInfo MakeInfo(const Surface& surface, PixelFormat format) {
    ImageInfo info{surface};
    info.format = format;
    return info;
}

/// The full-image copy CopyImage takes.
std::vector<ImageCopy> FullCopy() {
    return {ImageCopy{
        .src_subresource = SubresourceLayers{},
        .dst_subresource = SubresourceLayers{},
        .src_offset = {},
        .dst_offset = {},
        .extent = {64, 64, 1},
    }};
}

BlitConfig FullBlit(Tegra::Engines::Fermi2D::Filter filter, bool must_accelerate) {
    return BlitConfig{
        .operation = Tegra::Engines::Fermi2D::Operation::SrcCopy,
        .filter = filter,
        .must_accelerate = must_accelerate,
        .dst_x0 = 0,
        .dst_y0 = 0,
        .dst_x1 = 64,
        .dst_y1 = 64,
        .src_x0 = 0,
        .src_y0 = 0,
        .src_x1 = 64,
        .src_y1 = 64,
    };
}

} // namespace

TEST_CASE("A rejected 2D blit leaves a parked deferred clear for the next draw's load-op",
          "[video_core][clear_ordering]") {
    Harness h;
    // The game clears an MSAA colour attachment (parked: no render pass is active) and then
    // issues a 2D blit the texture cache rejects. The clear is still needed as the load-op of
    // the draw that loads the attachment, so the rejected blit must not realize it: realizing
    // it ends in a standalone clear-only pass whose MSAA store can be DONT_CARE
    // (Scheduler::RealizeDeferredClear picks the DiscardsMsaa* variants).
    constexpr GPUVAddr attachment_addr = 0x2000'0000;
    constexpr GPUVAddr scratch_addr = 0x3000'0000;
    const Surface attachment = MakeSurface(attachment_addr, 64, 64);
    const Surface scratch = MakeSurface(scratch_addr, 64, 64);
    ImageInfo attachment_info = MakeInfo(attachment, PixelFormat::R32_FLOAT);
    attachment_info.num_samples = 4; // the MSAA attachment the clear is parked on
    h.Insert(attachment_info, attachment_addr);
    h.Insert(MakeInfo(scratch, PixelFormat::R32_FLOAT), scratch_addr);

    int msaa_framebuffer{};
    h.log.Clear();
    h.scheduler.DeferColorClear(&msaa_framebuffer);

    // Rejected: nothing is GpuModified and the copy is not accelerated, so GetBlitImages
    // declines before BlitImage writes anything.
    REQUIRE_FALSE(h.cache.BlitImage(attachment, scratch, FullBlit(
        Tegra::Engines::Fermi2D::Filter::Point, false)));
    REQUIRE(h.log.Count(Event::FlushRequest) == 0);
    REQUIRE(h.scheduler.Pending());

    // The draw/load of the same attachment consumes the clear as its load-op.
    h.scheduler.RequestRenderpass(&msaa_framebuffer);
    CAPTURE(EventNames(h.log.events));
    REQUIRE(h.log.events == std::vector<Event>{Event::DeferColorClear, Event::RealizeLoadOp,
                                               Event::BeginPass});
}

TEST_CASE("CopyImage's early returns leave a parked deferred clear for the next draw's load-op",
          "[video_core][clear_ordering]") {
    Harness h;
    // CopyImage bails out before copying when the copy cannot land: unwritten sparse blocks on
    // either side, or a BC1/BC3 mismatch on re-compressed ASTC. Those returns write nothing, so
    // they must not realize a parked clear either (PR #185 review).
    constexpr GPUVAddr plain_addr = 0x2000'0000;
    constexpr GPUVAddr unwritten_addr = 0x3000'0000;
    constexpr GPUVAddr astc_opaque_addr = 0x4000'0000;
    constexpr GPUVAddr astc_alpha_addr = 0x5000'0000;
    const Surface plain_surface = MakeSurface(plain_addr, 64, 64);
    const ImageId plain = h.Insert(MakeInfo(plain_surface, PixelFormat::R32_FLOAT), plain_addr);
    const ImageId unwritten =
        h.Insert(MakeInfo(MakeSurface(unwritten_addr, 64, 64), PixelFormat::R32_FLOAT),
                 unwritten_addr);
    ImageInfo astc_opaque = MakeInfo(MakeSurface(astc_opaque_addr, 64, 64),
                                     PixelFormat::ASTC_2D_4X4_UNORM);
    astc_opaque.astc_opaque = true;
    ImageInfo astc_alpha = MakeInfo(MakeSurface(astc_alpha_addr, 64, 64),
                                    PixelFormat::ASTC_2D_4X4_UNORM);
    astc_alpha.astc_opaque = false;
    const ImageId opaque = h.Insert(astc_opaque, astc_opaque_addr);
    const ImageId alpha = h.Insert(astc_alpha, astc_alpha_addr);
    TextureCacheTestPeer::Image(h.cache, unwritten).flags |= ImageFlagBits::UnwrittenSparse;

    int framebuffer{};
    h.log.Clear();
    h.scheduler.DeferColorClear(&framebuffer);

    TextureCacheTestPeer::CopyImage(h.cache, unwritten, plain, FullCopy());
    TextureCacheTestPeer::CopyImage(h.cache, opaque, alpha, FullCopy());
    REQUIRE(h.log.Count(Event::FlushRequest) == 0);
    REQUIRE(h.log.Count(Event::Copy) == 0);
    REQUIRE(h.scheduler.Pending());

    h.scheduler.RequestRenderpass(&framebuffer);
    CAPTURE(EventNames(h.log.events));
    REQUIRE(h.log.events == std::vector<Event>{Event::DeferColorClear, Event::RealizeLoadOp,
                                               Event::BeginPass});
}

TEST_CASE("A valid 2D depth blit lands after the parked depth clear realizes",
          "[video_core][clear_ordering]") {
    Harness h;
    // The #101 case: Ocarina of Time clears the occlusion depth map, blits the half-res scene
    // depth over it and lets the terrain layer sample the map to hide itself behind the scene.
    // The clear is parked until the map's next render pass, which is never drawn: the copy
    // bypasses render passes entirely. Realizing the clear after the copy wipes it, so the copy
    // has to land after the realize. GetFramebufferId only flushes when it creates a
    // framebuffer, so from the second frame on the only ordering guarantee is the flush in
    // BlitImage itself (PR #185).
    constexpr GPUVAddr map_addr = 0x2000'0000;
    constexpr GPUVAddr scene_addr = 0x3000'0000;
    const Surface map_surface = MakeSurface(map_addr, 64, 64);
    const Surface scene_surface = MakeSurface(scene_addr, 64, 64);
    // The 2D engine describes the depth surfaces with a colour-like format; GetBlitImages
    // deduces the depth formats from the cached images (DeduceBlitImages).
    h.Insert(MakeInfo(map_surface, PixelFormat::D32_FLOAT), map_addr);
    h.Insert(MakeInfo(scene_surface, PixelFormat::D32_FLOAT), scene_addr);

    int map_framebuffer{};
    int layer_framebuffer{};
    for (int frame = 0; frame < 2; ++frame) {
        CAPTURE(frame);
        h.log.Clear();
        h.scheduler.DeferDepthStencilClear(&map_framebuffer);

        REQUIRE(h.cache.BlitImage(map_surface, scene_surface, FullBlit(
            Tegra::Engines::Fermi2D::Filter::Point, true)));
        REQUIRE(h.log.Count(Event::BlitCopy) == 1);

        // The clear must realize before the copy in every frame.
        const std::optional<size_t> realize = h.log.LastIndex(Event::RealizeStandalone);
        const std::optional<size_t> copy = h.log.FirstIndex(Event::BlitCopy);
        REQUIRE(realize.has_value());
        REQUIRE(copy.has_value());
        REQUIRE(*realize < *copy);
        if (frame > 0) {
            // The framebuffers are cached: only BlitImage's own flush can order the clear.
            REQUIRE(h.log.Count(Event::FlushRequest) == 1);
        }

        // The terrain layer samples the map: the copy is complete and nothing realizes late
        // into the sample's reads.
        const size_t sample_at = h.log.events.size();
        h.log.Record(Event::Sample);
        h.scheduler.RequestRenderpass(&layer_framebuffer);
        const std::optional<size_t> realize_after_sample = h.log.LastIndex(Event::RealizeStandalone);
        REQUIRE(realize_after_sample.has_value());
        REQUIRE(*realize_after_sample < sample_at);
    }
}

TEST_CASE("PrepareDmaImage flushes a parked clear only for a modifying DMA access",
          "[video_core][clear_ordering]") {
    Harness h;
    constexpr GPUVAddr addr = 0x2000'0000;
    const Surface surface = MakeSurface(addr, 64, 64);
    const ImageId image = h.Insert(MakeInfo(surface, PixelFormat::R32_FLOAT), addr);

    int framebuffer{};
    h.log.Clear();
    h.scheduler.DeferDepthStencilClear(&framebuffer);

    // A read-only access writes nothing: the clear stays parked for the next draw's load-op.
    TextureCacheTestPeer::PrepareDmaImage(h.cache, image, addr, false);
    CAPTURE(EventNames(h.log.events));
    REQUIRE(h.log.events == std::vector<Event>{Event::DeferDepthStencilClear});
    REQUIRE(h.scheduler.Pending());

    // The write realizes the clear before it lands.
    TextureCacheTestPeer::PrepareDmaImage(h.cache, image, addr, true);
    CAPTURE(EventNames(h.log.events));
    REQUIRE(h.log.events == std::vector<Event>{Event::DeferDepthStencilClear,
                                               Event::FlushRequest, Event::RealizeStandalone});
    REQUIRE_FALSE(h.scheduler.Pending());
}
