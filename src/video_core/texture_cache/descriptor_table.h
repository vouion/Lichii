// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename T, typename Id = u32>
class DescriptorTable {
public:
    [[nodiscard]] bool Synchronize(GPUVAddr gpu_addr, u32 limit) noexcept {
        bool ret = !(current_gpu_addr == gpu_addr && current_limit == limit);
        if (ret) {
            Refresh(gpu_addr, limit);
        }
        return ret;
    }

    void Invalidate() noexcept {
        // Cleared on the next read: the texture cache invalidates every channel's tables for
        // each image it deletes, dozens of times a frame while its collector runs.
        invalidated = true;
    }

    std::span<Id> LiveBoundIds() noexcept {
        return std::span{bound_ids}.first(
            (std::min)(bound_ids.size(), size_t{current_limit} + 1));
    }

    template <class MemoryManager>
    [[nodiscard]] std::pair<T, bool> Read(const MemoryManager& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        std::pair<T, bool> result;
        // The guest rewrites descriptor entries while this runs (scene loads stream new
        // textures into recycled slots, and fences release before GPU work commits). A read
        // that lands in the middle of a rewrite tears the entry, and a page with pending
        // GPU-written data reads stale; both surface as wrong-format, wrong-swizzle views
        // (the format=0 asserts, green channel loss, blinking UI and stretched textures).
        // Read twice and fall back to the flushing read when they disagree, so a torn or
        // stale entry resolves to the guest's real bytes instead of being bound. Both reads
        // share one translation of the address (HostPointer) but are still two independent
        // observations of guest memory: a rewrite landing between them is caught as before.
        const u8* const host = HostPointer(gpu_memory, gpu_addr);
        const auto read_fast = [&](T* out) noexcept {
            if (host) {
                std::memcpy(out, host, sizeof(T));
            } else {
                gpu_memory.ReadBlockUnsafe(gpu_addr, out, sizeof(T));
            }
        };
        read_fast(std::addressof(result.first));
        const u64 bit = 1ULL << (index % 64);
        if ((read_descriptors[index / 64] & bit) != 0 && !(result.first != descriptors[index])) {
            // The first observation already equals the entry this table has bound, so there is
            // nothing to re-resolve: repeat binds skip the second read and its torn-guard
            // compare. The second observation only exists to catch a rewrite (or a download
            // landing) between the two reads; skipping it means such an in-flight write is
            // picked up by the next Read of this index instead of this one - the entry was
            // still the bound value when the first read observed it, and any later read that
            // sees different bytes takes the full path below. A stale entry can therefore
            // never outlive one draw, unlike a serial-keyed skip with no change serial.
            result.second = false;
            return result;
        }
        T again;
        read_fast(std::addressof(again));
        if (std::memcmp(&result.first, &again, sizeof(T)) != 0) {
            gpu_memory.ReadBlock(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if ((read_descriptors[index / 64] & bit) != 0) {
            result.second = result.first != descriptors[index];
        } else {
            read_descriptors[index / 64] |= bit;
            result.second = true;
        }
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
    }

    void Refresh(GPUVAddr gpu_addr, u32 limit) noexcept {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        ClearReadDescriptors();
        invalidated = false;
        //
        descriptors.resize(num_descriptors);
        bound_ids.resize(num_descriptors, Id{0});
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    std::vector<Id> bound_ids;
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
    bool invalidated{}; ///< read_descriptors is to be cleared before the next read

private:
    // This runs for every texture and sampler of every draw. An entry within one page is
    // contiguous in host memory, so a single page lookup replaces the generic block read.
    // The lookup itself is cached per page: the translation of a page only changes when the
    // mapping does, and every mapping change bumps MemoryManager::MapSerial (and is a new
    // MemoryManager when GetID changes), so a cached pointer is reused only while both still
    // hold. Memory managers without those (the tests' mocks) translate every read.
    template <class MemoryManager>
    [[nodiscard]] const u8* HostPointer(const MemoryManager& gpu_memory,
                                        GPUVAddr gpu_addr) noexcept {
        if ((gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) > Core::Memory::YUZU_PAGESIZE) {
            return nullptr; // The entry spans two pages: use the generic block read.
        }
        if constexpr (requires {
                          gpu_memory.MapSerial();
                          gpu_memory.GetID();
                      }) {
            const GPUVAddr page = gpu_addr & ~GPUVAddr{Core::Memory::YUZU_PAGEMASK};
            const u64 serial = gpu_memory.MapSerial();
            HostPage& entry = host_pages[(page >> Core::Memory::YUZU_PAGEBITS) & 3];
            if (entry.page != page || entry.serial != serial || entry.id != gpu_memory.GetID()) {
                entry = {page, serial, gpu_memory.GetID(), gpu_memory.GetPointer(page)};
            }
            return entry.host ? entry.host + (gpu_addr & Core::Memory::YUZU_PAGEMASK) : nullptr;
        }
        return gpu_memory.GetPointer(gpu_addr);
    }

    void ClearReadDescriptors() noexcept {
        // Reserved capacity is not addressable. A later limit increase clears its new range.
        const size_t words = (std::min)(read_descriptors.size(), size_t{current_limit} / 64 + 1);
        std::fill_n(read_descriptors.begin(), words, u64{0});
        const size_t ids = (std::min)(bound_ids.size(), size_t{current_limit} + 1);
        std::fill_n(bound_ids.begin(), ids, Id{0});
    }

    /// The host pointer of a 4 KiB page of the pool, kept while the mapping stands.
    struct HostPage {
        GPUVAddr page{};
        u64 serial{};
        std::size_t id{};
        const u8* host{};
    };
    std::array<HostPage, 4> host_pages{};
};

} // namespace VideoCommon
