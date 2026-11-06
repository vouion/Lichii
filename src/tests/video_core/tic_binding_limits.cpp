// SPDX-License-Identifier: GPL-3.0-or-later

// Regression tests for garbage TICs at the texture-cache binding seam (crash-216: OoT at
// gameplay start fed zero-format TICs to the format lookup and asserted in
// format_lookup_table.cpp). A TIC entry the game never wrote is unbound: the binding seam must
// hand out the null view instead of decoding its zero format through ImageInfo/ImageViewInfo
// and building a bogus image out of garbage dimensions.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/device_memory.h"
#include "video_core/textures/texture.h"
#include "video_core/texture_cache/util.h"

#include "tests/video_core/texture_cache_test_context.h"

namespace {

using TextureCacheTests::Harness;
using Tegra::Texture::ComponentType;
using Tegra::Texture::TICEntry;
using Tegra::Texture::TICHeaderVersion;
using Tegra::Texture::TextureFormat;
using Tegra::Texture::TextureType;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewInOut;
using VideoCommon::IsValidEntry;
using VideoCommon::NULL_IMAGE_VIEW_ID;

constexpr GPUVAddr ImageAddr = 0x2000'0000;
constexpr GPUVAddr PoolAddr = 0x4000'0000;
constexpr GPUVAddr UnmappedAddr = 0x3000'0000;

/// A small block-linear 2D TIC, as the game would write it.
TICEntry MakeTic(GPUVAddr address) {
    TICEntry entry{};
    entry.format.Assign(TextureFormat::A8B8G8R8);
    entry.r_type.Assign(ComponentType::UNORM);
    entry.g_type.Assign(ComponentType::UNORM);
    entry.b_type.Assign(ComponentType::UNORM);
    entry.a_type.Assign(ComponentType::UNORM);
    entry.texture_type.Assign(TextureType::Texture2D);
    entry.header_version.Assign(TICHeaderVersion::BlockLinear);
    entry.block_width.Assign(0);
    entry.block_height.Assign(4);
    entry.block_depth.Assign(0);
    entry.width_minus_one.Assign(63);
    entry.height_minus_1.Assign(63);
    entry.depth_minus_1.Assign(0);
    entry.max_mip_level.Assign(0);
    entry.address_low = static_cast<u32>(address);
    entry.address_high.Assign(static_cast<u32>(address >> 32));
    return entry;
}

/// The unbound entry a game leaves behind while rewriting its TIC table.
TICEntry MakeUnboundTic(GPUVAddr address) {
    TICEntry entry = MakeTic(address);
    entry.format.Assign(TextureFormat{});
    return entry;
}

void MapFixture(Harness& h) {
    h.MapGuest(ImageAddr, 0x20'0000);
    h.MapGuest(PoolAddr, 0x1'0000);
}

} // namespace

TEST_CASE("Unbound TIC entries (zero format) are rejected as invalid",
          "[video_core][tic_binding]") {
    Harness h;
    MapFixture(h);

    // An entry the game never wrote at all.
    REQUIRE_FALSE(IsValidEntry(h.memory, TICEntry{}));

    // The crash-216 case: format=0 with a mapped address used to pass the validity check and
    // decode its zero format through ImageInfo on the way to CreateImageView.
    REQUIRE_FALSE(IsValidEntry(h.memory, MakeUnboundTic(ImageAddr)));

    // The same entry at an unmapped address used to trip the format lookup inside the validity
    // check's own ImageInfo size fallback.
    REQUIRE_FALSE(IsValidEntry(h.memory, MakeUnboundTic(UnmappedAddr)));

    // A real entry stays valid when its image is mapped, and rejected when it is not.
    REQUIRE(IsValidEntry(h.memory, MakeTic(ImageAddr)));
    REQUIRE_FALSE(IsValidEntry(h.memory, MakeTic(UnmappedAddr)));

    // An address outside the GPU address space is rejected either way.
    REQUIRE_FALSE(IsValidEntry(h.memory, MakeTic(1ULL << 40)));
    REQUIRE_FALSE(IsValidEntry(h.memory, MakeUnboundTic(1ULL << 40)));
}

TEST_CASE("Zero-format TICs and out-of-range handles stay unbound at the descriptor seam",
          "[video_core][tic_binding]") {
    Harness h;
    MapFixture(h);

    // Two entries in the bound table: a real texture and one the game has not written yet.
    const TICEntry entries[2] = {MakeTic(ImageAddr), MakeUnboundTic(ImageAddr + 0x8'0000)};
    h.memory.WriteBlock(PoolAddr, entries, sizeof(entries));
    auto& regs = h.channel.payload->maxwell_3d.regs;
    regs.tex_header.address_high = static_cast<u32>(PoolAddr >> 32);
    regs.tex_header.address_low = static_cast<u32>(PoolAddr);
    regs.tex_header.limit = 1;
    h.cache.SynchronizeDescriptors(false);

    std::vector<ImageViewInOut> views(3);
    views[0].index = 0; // the real texture
    views[1].index = 1; // the unbound entry, within the bound range
    views[2].index = 2; // a handle past the bound range
    h.cache.FillImageViews(views, false, false);

    REQUIRE(views[0].id != NULL_IMAGE_VIEW_ID);
    REQUIRE(views[1].id == NULL_IMAGE_VIEW_ID);
    REQUIRE(views[2].id == NULL_IMAGE_VIEW_ID);
}

TEST_CASE("A TIC the guest rewrites between draws re-resolves the binding",
          "[video_core][tic_binding]") {
    Harness h;
    MapFixture(h);

    constexpr GPUVAddr SecondImageAddr = ImageAddr + 0x10'0000;
    const TICEntry original = MakeTic(ImageAddr);
    h.memory.WriteBlock(PoolAddr, &original, sizeof(original));
    auto& regs = h.channel.payload->maxwell_3d.regs;
    regs.tex_header.address_high = static_cast<u32>(PoolAddr >> 32);
    regs.tex_header.address_low = static_cast<u32>(PoolAddr);
    regs.tex_header.limit = 0;
    h.cache.SynchronizeDescriptors(false);

    std::vector<ImageViewInOut> views(1);
    views[0].index = 0;
    h.cache.FillImageViews(views, false, false);
    const ImageViewId first = views[0].id;
    REQUIRE(first != NULL_IMAGE_VIEW_ID);

    // The guest rewrites the entry for a new texture at the same slot (scene load): the next
    // draw must not reuse the previous scene's view.
    const TICEntry rewritten = MakeTic(SecondImageAddr);
    h.memory.WriteBlock(PoolAddr, &rewritten, sizeof(rewritten));
    h.cache.FillImageViews(views, false, false);
    REQUIRE(views[0].id != NULL_IMAGE_VIEW_ID);
    REQUIRE(views[0].id != first);

    // Unwritten again: unbound, not a stale view.
    const TICEntry unwritten = MakeUnboundTic(SecondImageAddr);
    h.memory.WriteBlock(PoolAddr, &unwritten, sizeof(unwritten));
    h.cache.FillImageViews(views, false, false);
    REQUIRE(views[0].id == NULL_IMAGE_VIEW_ID);

    // The same bytes resolve to the same view again (the entry-keyed cache is content-keyed).
    h.memory.WriteBlock(PoolAddr, &original, sizeof(original));
    h.cache.FillImageViews(views, false, false);
    REQUIRE(views[0].id == first);
}
