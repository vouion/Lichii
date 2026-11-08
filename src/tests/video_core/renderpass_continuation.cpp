// SPDX-License-Identifier: GPL-3.0-or-later

// The continuation rule behind Scheduler::RequestRenderpass: a draw stays in the pass it is in
// only while that pass is open for the same framebuffer and area. Matching the target alone is
// what the first clear-variant continuation did, and after a query copy, upload or dispatch had
// ended the pass the next draw was then recorded with no render pass at all (washed frames).

#include <cstdint>
#include <cstring>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace {

using Vulkan::CanContinueRenderPass;
using Vulkan::RenderPassBinding;

/// A distinct non-null handle whatever the platform's handle representation is.
template <typename Handle>
Handle DummyHandle(std::uintptr_t tag) {
    Handle handle{};
    const auto value = static_cast<std::uintptr_t>(tag);
    std::memcpy(&handle, &value, sizeof(Handle) < sizeof(value) ? sizeof(Handle) : sizeof(value));
    return handle;
}

/// The pass `framebuffer_key`'s object has bound: the handle and the key move together.
RenderPassBinding Pass(VkRenderPass renderpass, const void* framebuffer_key, u32 width = 640,
                       u32 height = 480, bool load_variant_only = false) {
    const auto key = reinterpret_cast<std::uintptr_t>(framebuffer_key);
    return RenderPassBinding{
        .renderpass = renderpass,
        .framebuffer = DummyHandle<VkFramebuffer>(key),
        .framebuffer_key = framebuffer_key,
        .render_area = {width, height},
        .load_variant_only = load_variant_only,
    };
}

const VkRenderPass BasePass = DummyHandle<VkRenderPass>(1);
const VkRenderPass ClearVariant = DummyHandle<VkRenderPass>(2);

} // Anonymous namespace

TEST_CASE("A draw never continues a render pass that is not open",
          "[video_core][renderpass_continuation]") {
    const int target{};
    const RenderPassBinding requested = Pass(BasePass, &target);
    // EndRenderPass clears the pass but leaves the framebuffer behind in the scheduler state: a
    // continuation check that ignores that would record the draw with no render pass at all.
    REQUIRE_FALSE(CanContinueRenderPass(RenderPassBinding{}, requested));
    // The stale target of a pass that has ended is not an open pass either.
    RenderPassBinding ended = Pass(BasePass, &target);
    ended.renderpass = VkRenderPass{};
    REQUIRE_FALSE(CanContinueRenderPass(ended, requested));
}

TEST_CASE("A draw stays in the pass bound to the same target", "[video_core][renderpass_continuation]") {
    const int target{};
    const int other_target{};
    const RenderPassBinding open = Pass(BasePass, &target);
    CHECK(CanContinueRenderPass(open, Pass(BasePass, &target)));
    // A different framebuffer, or the same handle under a different framebuffer object, restarts.
    CHECK_FALSE(CanContinueRenderPass(open, Pass(BasePass, &other_target)));
    CHECK_FALSE(CanContinueRenderPass(open, Pass(ClearVariant, &other_target)));
    // A different render area restarts too (the framebuffer's size changed under it).
    CHECK_FALSE(CanContinueRenderPass(open, Pass(BasePass, &target, 1280, 720)));
}

TEST_CASE("A deferred clear's pass holds the draws after it", "[video_core][renderpass_continuation]") {
    const int target{};
    // The clear variant's loadOps ran at begin, so the draws that follow continue it instead of
    // re-LOADing the surface the clear just wrote.
    const RenderPassBinding clearing = Pass(ClearVariant, &target, 640, 480, true);
    CHECK(CanContinueRenderPass(clearing, Pass(BasePass, &target)));
    CHECK(CanContinueRenderPass(clearing, Pass(ClearVariant, &target)));
    // A variant that discards stores (the MSAA scratch form) changes a storeOp, so it never holds
    // draws: the base pass' STORE has to come back.
    const RenderPassBinding discarding = Pass(ClearVariant, &target, 640, 480, false);
    CHECK_FALSE(CanContinueRenderPass(discarding, Pass(BasePass, &target)));
}
