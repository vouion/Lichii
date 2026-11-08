// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <type_traits>

#include "common/common_types.h"
#include "common/logging.h"
#include "core/device_memory_manager.h"

namespace VideoCommon {

/// What a buffer cache runtime offers for stream uniform reuse.
///  - StreamUniformRef: a small trivially copyable description of one stream allocation (the
///    host buffer, its offset and whatever else the runtime needs to add it to a descriptor
///    payload again, e.g. the device address of the descriptor buffer path).
///  - LastStreamUniformRef(): the allocation made by the latest BindMappedUniformBuffer call.
///  - RebindStreamUniform(ref, size): adds `ref` to the descriptor payload exactly as
///    BindMappedUniformBuffer added the allocation the first time.
template <class R>
concept StreamUniformReusable = requires(R& runtime, u32 size) {
    typename R::StreamUniformRef;
    requires std::is_trivially_copyable_v<typename R::StreamUniformRef>;
    { runtime.LastStreamUniformRef() } -> std::same_as<typename R::StreamUniformRef>;
    runtime.RebindStreamUniform(runtime.LastStreamUniformRef(), size);
    { runtime.CurrentTick() } -> std::convertible_to<u64>;
};

/// EDEN_NX2_UBOREUSE=0 turns the reuse off (every small uniform buffer is copied, as before).
[[nodiscard]] inline bool UniformStreamReuseEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_UBOREUSE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_GUESTGEN=1 opts into trusting the guest-write generations to skip a small uniform's
/// copy+compare (and a buffer's upload walk). Off by default: the generations are bumped only by
/// the buffer cache's own entry points, so a writer that reaches guest RAM without going through
/// them (a texture-cache download lands via MemoryManager::WriteBlockUnsafe and bumps nothing) can
/// change bytes the table still calls unchanged, and the skip then serves a stale uniform - wrong
/// exposure/lighting, a washed-out image. Correctness first: copy and compare unless this is
/// explicitly turned on. The reuse's own read+memcmp path is unaffected and stays on; this only
/// controls skipping that read on a generation proof. Callers pass it as `gen_skip_ok`.
[[nodiscard]] inline bool GuestGenSkipEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_GUESTGEN");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// Temporary measurement counters for the generation-keyed skip (rate-limited log below).
struct UniformStreamSkipStats {
    u64 attempts{};      ///< binds the generations proved unchanged
    u64 skips{};         ///< of those, the copy+compare actually skipped
    u64 copies{};        ///< gen-tracked binds that read and compared guest memory
    u64 audits{};        ///< sampled skips that also copied to check the proof
    u64 audit_failed{};  ///< audits that found changed bytes: a writer the generations missed
};
[[nodiscard]] inline UniformStreamSkipStats& UniformStreamSkipCounters() noexcept {
    static UniformStreamSkipStats stats;
    return stats;
}

/// Set when a sampled audit catches guest bytes changing without a generation bump: every skip
/// falls back to the copy+compare for the rest of the run.
[[nodiscard]] inline bool& UniformStreamSkipBroken() noexcept {
    static bool broken = false;
    return broken;
}

/// Rate-limited totals so a run can measure the skip without a profiler.
inline void LogUniformStreamSkip(const char* event) {
    static u64 logged{};
    const UniformStreamSkipStats& stats = UniformStreamSkipCounters();
    const u64 total = stats.skips + stats.copies;
    if (logged != 0 && (total & 0x3FFFF) != 0) {
        return;
    }
    ++logged;
    LOG_WARNING(HW_GPU,
                "Uniform stream gen skip {}: skips={} copies={} audits={} audit_failed={} "
                "broken={}",
                event, stats.skips, stats.copies, stats.audits, stats.audit_failed,
                UniformStreamSkipBroken());
}

template <class Ref>
class UniformStreamReuse {
public:
    static constexpr u32 MaxBytes = 4096; // BufferCache's DEFAULT_SKIP_CACHE_SIZE
    static constexpr size_t NumStages = 5;
    static constexpr size_t NumSlots = 18; // Maxwell constant buffers per stage
    static constexpr size_t MaxGenPages = 4; // GuestWriteGenerations::MAX_SNAPSHOT_PAGES

    struct Slot {
        Ref ref{};
        u64 tick{};
        u32 size{}; // 0: nothing streamed yet
        u64 addr{}; // device address of the bytes, for the generation key
        u8 gen_pages{}; // 0: no generation snapshot to skip from
        std::array<u64, MaxGenPages> gens{};
        std::array<u8, MaxBytes> bytes{};
    };

    void Reset() noexcept {
        for (auto& stage : slots) {
            for (Slot& slot : stage) {
                slot.size = 0;
                slot.tick = 0;
                slot.gen_pages = 0;
            }
        }
    }

    std::array<std::array<Slot, NumSlots>, NumStages> slots{};
    std::array<u8, MaxBytes> scratch{};
};

template <class Ref>
[[nodiscard]] inline UniformStreamReuse<Ref>& UniformStreamReuseState() noexcept {
    static UniformStreamReuse<Ref> state;
    return state;
}

/// Forgets every slot. Call it when a buffer cache (and so its runtime, scheduler and stream
/// buffer) is created: ticks restart with a new scheduler and the old references are dead.
template <class Runtime>
inline void ResetUniformStreamReuse() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().Reset();
    }
}

/// Binds the small uniform buffer at `device_addr` for (stage, cbuf `index`) through the stream
/// buffer, reusing the slot's previous allocation when the bytes and the command buffer are the
/// same. Returns false (nothing bound) when the buffer does not qualify, so the caller binds it
/// the classic way.
///
/// `guest_gen` snapshots and compares per-page guest write generations (buffer_cache_base.h's
/// GuestWriteGenerations): with `gen_skip_ok`, a range whose pages hold the snapshot of the
/// slot's last copy is proven unchanged, and neither the copy nor the compare runs. Any doubt
/// falls back to reading and comparing guest memory.
template <class Runtime, class DeviceMemory, class GuestGen>
[[nodiscard]] bool BindSmallUniformReusing(Runtime& runtime, DeviceMemory& device_memory,
                                           GuestGen& guest_gen, bool gen_skip_ok, size_t stage,
                                           u32 index, u32 binding_index, u64 device_addr,
                                           u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    u8* const scratch = reuse.scratch.data();
    UniformStreamSkipStats& stats = UniformStreamSkipCounters();

    const bool gen_trusted =
        gen_skip_ok && !UniformStreamSkipBroken() && slot.size == size &&
        slot.addr == device_addr && guest_gen.Unchanged(device_addr, size, slot.gens, slot.gen_pages);
    bool same_bytes = false;
    bool have_scratch = false;
    if (gen_trusted) {
        ++stats.attempts;
        same_bytes = true;
        if ((stats.attempts & 0xFF) == 0) {
            // Sampled audit: a mismatch means a guest writer the generations do not see.
            ++stats.audits;
            device_memory.ReadBlockUnsafe(device_addr, scratch, size);
            have_scratch = true;
            if (std::memcmp(slot.bytes.data(), scratch, size) != 0) {
                ++stats.audit_failed;
                UniformStreamSkipBroken() = true;
                same_bytes = false;
                LogUniformStreamSkip("audit-failed");
            }
        }
    }
    // The guest bytes themselves: read them once and compare them in place, so the equal case
    // (the common one) pays no intermediate copy and a difference copies straight from the
    // same pointer the compare read. A range that crosses a page boundary, or a memory manager
    // without host pointers (the tests'), falls back to the copy-then-compare it did before.
    // Nothing is skipped here: the same bytes are observed either way.
    const u8* guest = have_scratch ? scratch : nullptr;
    if (!same_bytes && guest == nullptr) {
        if constexpr (requires { device_memory.template GetPointer<u8>(u64{}); }) {
            if (size <= Core::DEVICE_PAGESIZE - (device_addr & Core::DEVICE_PAGEMASK)) {
                guest = device_memory.template GetPointer<u8>(device_addr);
            }
        }
        if (guest == nullptr) {
            device_memory.ReadBlockUnsafe(device_addr, scratch, size);
            guest = scratch;
        }
    }
    if (!same_bytes) {
        same_bytes = slot.size == size && std::memcmp(guest, slot.bytes.data(), size) == 0;
        ++stats.copies;
        // The rate-limited log moved off the hot path: it used to run on every compare.
        if (stats.copies == 1 || (stats.copies & 0x3FFFF) == 0) {
            LogUniformStreamSkip("copy");
        }
    } else {
        ++stats.skips;
    }
    if (gen_skip_ok) {
        // The guest bytes are observed above; the proven-unchanged slot bytes are the guest's.
        slot.addr = device_addr;
        slot.gen_pages = guest_gen.Snapshot(device_addr, size, slot.gens);
    } else {
        slot.gen_pages = 0;
    }
    if (same_bytes && slot.tick == runtime.CurrentTick()) {
        runtime.RebindStreamUniform(slot.ref, size);
        return true;
    }
    const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
    if (same_bytes) {
        std::memcpy(span.data(), slot.bytes.data(), size);
    } else {
        // One observation of the guest's bytes feeds both the stream allocation and the slot's
        // copy of them: caching the stream's own contents keeps the two identical even if the
        // guest writes again between the two copies (which is what made the old scratch
        // snapshot necessary).
        std::memcpy(span.data(), guest, size);
        slot.size = size;
        std::memcpy(slot.bytes.data(), span.data(), size);
    }
    slot.ref = runtime.LastStreamUniformRef();
    // After the allocation: the tick the stream region is tagged with is the one current now.
    slot.tick = runtime.CurrentTick();
    return true;
}

} // namespace VideoCommon
