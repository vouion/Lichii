// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <memory>

#include "common/common_types.h"

/// Lock-free coverage filter for the guest CPU write path (Memory::HandleRasterizerWrite).
/// A store to GPU-registered guest memory notifies the texture and shader caches so a texture
/// or shader compiled from those bytes cannot outlive a CPU edit. For ring-buffer stores the
/// notification always finds nothing, but it still takes the texture cache's and the shader
/// cache's global mutexes - thousands of times per frame. This filter answers "can any texture
/// or shader cover this range?" with one or two relaxed loads and lets the store skip those
/// locks when the answer is no.
///
/// Correctness (why a skip cannot lose a notification):
/// - The bitmap is monotonic: bits are only ever set, at the same moment the covering entry
///   lands in the cache's own page table (texture page_table, shader invalidation_cache,
///   rejected_native), and before that entry can be observed. A clear bit therefore proves
///   the range was never covered, and the notification would be a no-op anyway.
/// - The store reads the bitmap twice: once before the write (in HandleRasterizerWrite) and
///   once after the write lands (Write<T>). Today's notification runs between those two reads
///   under the cache mutex; a registration that completed before it is found by the first
///   read (the mark precedes the insert), and one that lands between the reads is found by
///   the second. Every interleaving today's mutex-ordered notify hits, this filter also
///   catches; it can only catch more.
/// - A set bit just means "maybe": the caller then does exactly what it did before, so false
///   positives (colliding pages, eviction, monotonic saturation) cost time, never correctness.
///
/// EDEN_NX2_WRITE_FILTER=0 restores the previous behaviour: Covers() always returns true and
/// every store takes the locks.
namespace VideoCore::Nx2WriteFilter {

/// YUZU_PAGEBITS and DEVICE_PAGEBITS are both 12; the filter keys every structure on that
/// same4 KiB page number (texture page_table, shader invalidation_cache, store DAddr).
constexpr u64 PageBits = 12;
/// 16 GiB of guest address space = 4 M pages = 512 KiB of bits. Pages beyond it report
/// covered, which only costs the optimization.
constexpr u64 MaxPages = 1ULL << 22;

[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_WRITE_FILTER");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

class Coverage {
public:
    /// True when a texture or shader may cover [addr, addr + size): the caller must do the
    /// full notification. False proves nothing ever covered it, so the notification would be
    /// a no-op and may be skipped (and re-checked after the write lands).
    [[nodiscard]] bool Covers(u64 addr, u64 size) const noexcept {
        if (!Enabled()) [[unlikely]] {
            return true;
        }
        const u64 first = addr >> PageBits;
        const u64 last = (addr + size - 1) >> PageBits;
        for (u64 page = first;; ++page) {
            if (page >= MaxPages) [[unlikely]] {
                return true;
            }
            if (words[page >> 6].load(std::memory_order_acquire) & (1ULL << (page & 63))) {
                return true;
            }
            if (page == last) {
                return false;
            }
        }
    }

    /// Records that [addr, addr + size) is (now or already) covered by a texture or shader.
    /// Called from the cache's own registration path, before its covering entry is inserted,
    /// under the same mutex the notification takes. Never cleared: an unmarked page would
    /// only happen if a registration path forgot to call this, so each of them calls it for
    /// exactly the range it inserts.
    void Mark(u64 addr, u64 size) noexcept {
        if (!Enabled()) [[unlikely]] {
            return;
        }
        u64 page = addr >> PageBits;
        const u64 last = (addr + size - 1) >> PageBits;
        while (page <= last && page < MaxPages) {
            const u64 word = page >> 6;
            u64 mask = 0;
            do {
                mask |= 1ULL << (page & 63);
                ++page;
            } while (page <= last && page < MaxPages && (page >> 6) == word);
            words[word].fetch_or(mask, std::memory_order_release);
        }
    }

private:
    std::unique_ptr<std::atomic<u64>[]> words{
        std::make_unique<std::atomic<u64>[]>(MaxPages / 64)};
};

/// The process-wide filter. An inline function's local static is a single instance no matter
/// how many translation units use it.
[[nodiscard]] inline Coverage& Get() {
    static Coverage coverage;
    return coverage;
}

} // namespace VideoCore::Nx2WriteFilter
