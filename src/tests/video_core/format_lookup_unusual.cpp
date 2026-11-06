// SPDX-License-Identifier: GPL-3.0-or-later

// Regression tests for real-but-unusual TIC decodes (Task L: fog LUTs, water ripples and
// canopy lightmaps sampled garbage because their formats fell through the format lookup to
// the unimplemented fallback, which decodes everything as A8B8G8R8_UNORM). The forced-FP16
// component types hold UNORM/SNORM data sampled rounded to FP16 (sw_blitter/converter.cpp) and
// X8B8G8R8 is A8B8G8R8 with an unused alpha: both must decode to the base type's format so the
// channels and the bytes-per-block math stay right.

#include <catch2/catch_test_macros.hpp>

#include "video_core/surface.h"
#include "video_core/texture_cache/format_lookup_table.h"

namespace {

using Tegra::Texture::ComponentType;
using Tegra::Texture::TextureFormat;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::PixelFormat;

constexpr bool SRGB = true;

} // namespace

TEST_CASE("Forced-FP16 component types decode as their UNORM/SNORM base formats",
          "[video_core][format_lookup]") {
    using VideoCommon::PixelFormatFromTextureInfo;
    // UNORM_FORCE_FP16 on a 16-bit-per-channel format must not fall through to the
    // A8B8G8R8_UNORM fallback: that decodes 4 bytes per block instead of 8 and remaps every
    // channel (the green water ripples and canopy wash).
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::R16G16B16A16, ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       false) == PixelFormat::R16G16B16A16_UNORM);
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::R16G16B16A16, ComponentType::SNORM_FORCE_FP16,
                                       ComponentType::SNORM_FORCE_FP16,
                                       ComponentType::SNORM_FORCE_FP16,
                                       ComponentType::SNORM_FORCE_FP16,
                                       false) == PixelFormat::R16G16B16A16_SNORM);
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::R16G16, ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       false) == PixelFormat::R16G16_UNORM);

    // A mix of forced and plain components decodes like its base types.
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::R16G16B16A16, ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM, ComponentType::UNORM,
                                       ComponentType::UNORM,
                                       false) == PixelFormat::R16G16B16A16_UNORM);

    // The sRGB flag survives the normalization.
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::A8B8G8R8, ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       ComponentType::UNORM_FORCE_FP16,
                                       SRGB) == PixelFormat::A8B8G8R8_SRGB);
}

TEST_CASE("X8B8G8R8 decodes as A8B8G8R8 with its component types",
          "[video_core][format_lookup]") {
    using VideoCommon::PixelFormatFromTextureInfo;
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::X8B8G8R8, ComponentType::UNORM,
                                       ComponentType::UNORM, ComponentType::UNORM,
                                       ComponentType::UNORM,
                                       false) == PixelFormat::A8B8G8R8_UNORM);
    // Not the fallback's UNORM: the component type still matters.
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::X8B8G8R8, ComponentType::SNORM,
                                       ComponentType::SNORM, ComponentType::SNORM,
                                       ComponentType::SNORM,
                                       false) == PixelFormat::A8B8G8R8_SNORM);
    REQUIRE(PixelFormatFromTextureInfo(TextureFormat::X8B8G8R8, ComponentType::UNORM,
                                       ComponentType::UNORM, ComponentType::UNORM,
                                       ComponentType::UNORM,
                                       SRGB) == PixelFormat::A8B8G8R8_SRGB);
}

TEST_CASE("Unusual decodes keep the bytes-per-block the layout math needs",
          "[video_core][format_lookup]") {
    // A wrong bytes-per-block breaks every size/stride computation downstream (the stretched
    // UI texture class of the corruption).
    REQUIRE(BytesPerBlock(PixelFormat::R16G16B16A16_UNORM) == 8);
    REQUIRE(BytesPerBlock(PixelFormat::R16G16B16A16_SNORM) == 8);
    REQUIRE(BytesPerBlock(PixelFormat::R16G16_UNORM) == 4);
    REQUIRE(BytesPerBlock(PixelFormat::A8B8G8R8_UNORM) == 4);
}
