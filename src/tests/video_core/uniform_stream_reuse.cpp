// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"

namespace {
void Check(bool condition, const char* message) {
    INFO(message);
    REQUIRE(condition);
}

struct Memory {
    std::array<u8, 8192> bytes{};
    void ReadBlockUnsafe(u64 address, void* destination, u32 size) {
        Check(address + size <= bytes.size(), "read outside guest memory");
        std::memcpy(destination, bytes.data() + address, size);
    }
};

// A guest write, as the buffer cache sees it: the bytes change and the generations bump.
void Write(Memory& memory, VideoCommon::GuestWriteGenerations& gen, u64 address, u32 size) {
    memory.bytes[address] ^= u8{0x5a};
    gen.NoteWrite(address, size);
}

struct Runtime {
    struct StreamUniformRef {
        size_t allocation;
    };
    std::vector<std::vector<u8>> allocations;
    StreamUniformRef last{}, bound{};
    u64 tick{1};
    u64 CurrentTick() const {
        return tick;
    }
    StreamUniformRef LastStreamUniformRef() const {
        return last;
    }
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        allocations.emplace_back(size);
        last = bound = StreamUniformRef{allocations.size() - 1};
        return allocations.back();
    }
    void RebindStreamUniform(StreamUniformRef ref, u32 size) {
        Check(ref.allocation < allocations.size(), "reused a recycled allocation");
        Check(allocations[ref.allocation].size() == size, "reused wrong allocation size");
        bound = ref;
    }
    void NextTick() {
        ++tick;
        allocations.clear();
    }
};

void CheckBound(const Runtime& runtime, const Memory& memory, u64 address, u32 size) {
    const auto& bytes = runtime.allocations.at(runtime.bound.allocation);
    Check(bytes.size() == size &&
              std::memcmp(bytes.data(), memory.bytes.data() + address, size) == 0,
          "bound uniform data differs from guest memory");
}
} // namespace

TEST_CASE("Uniform stream reuse preserves data through mutation and session resets",
          "[video_core]") {
    using VideoCommon::BindSmallUniformReusing;
    Memory memory;
    Runtime runtime;
    VideoCommon::GuestWriteGenerations gen;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    auto bind = [&](size_t stage, u32 slot, u64 address, u32 size) {
        Check(BindSmallUniformReusing(runtime, memory, gen, true, stage, slot, slot, address, size),
              "valid small uniform rejected");
        CheckBound(runtime, memory, address, size);
    };
    bind(0, 0, 0, 64);
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 1, "identical bytes were copied twice");
    Write(memory, gen, 7, 64);
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 2, "changed bytes reused stale data");
    bind(0, 1, 0, 64);
    bind(1, 0, 0, 64);
    Check(runtime.allocations.size() == 4, "stage or slot shared the wrong allocation");
    runtime.NextTick();
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 1, "a new command buffer reused an old allocation");
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 2, "session reset retained an old allocation");
    bind(4, 17, 0, 4096);
    const auto before = runtime.allocations.size();
    for (auto [stage, slot, size] :
         {std::array<u32, 3>{5, 0, 64}, {0, 18, 64}, {0, 0, 0}, {0, 0, 4097}}) {
        Check(!BindSmallUniformReusing(runtime, memory, gen, true, stage, slot, slot, 0, size),
              "invalid uniform range did not use the fallback");
    }
    Check(runtime.allocations.size() == before, "fallback allocated stream memory");

    u32 random = 17;
    for (u32 index = 0; index < 10000; ++index) {
        random = random * 1664525U + 1013904223U;
        if (index % 37 == 0)
            runtime.NextTick();
        if (index % 503 == 0)
            VideoCommon::ResetUniformStreamReuse<Runtime>();
        const u32 size = 1 + random % 4096;
        const u64 address = (random >> 12) % (memory.bytes.size() - 4096);
        if (index % 3 == 0)
            Write(memory, gen, address + random % size, size);
        const size_t stage = (random >> 20) % 5;
        const u32 slot = (random >> 24) % 18;
        bind(stage, slot, address, size);
        bind(stage, slot, address, size);
    }
}

TEST_CASE("Uniform stream generation skip copies only on guest writes", "[video_core]") {
    using VideoCommon::BindSmallUniformReusing;
    Memory memory;
    Runtime runtime;
    VideoCommon::GuestWriteGenerations gen;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    auto bind = [&](u64 address, u32 size) {
        Check(BindSmallUniformReusing(runtime, memory, gen, true, 0, 0, 0, address, size),
              "valid small uniform rejected");
        CheckBound(runtime, memory, address, size);
    };
    bind(0, 128);
    bind(0, 128);
    Check(runtime.allocations.size() == 1, "unchanged bytes allocated twice");
    // Proven unchanged across a command buffer: a fresh allocation is filled from the slot's
    // bytes without reading guest memory.
    runtime.NextTick();
    bind(0, 128);
    Check(runtime.allocations.size() == 1, "unchanged bytes copied into the new command buffer");
    bind(0, 128);
    Check(runtime.allocations.size() == 1, "the new allocation was not reused");
    // A guest write bumps the generations, so the next bind copies again.
    Write(memory, gen, 8, 128);
    bind(0, 128);
    Check(runtime.allocations.size() == 2, "a written range skipped its copy");
    bind(0, 128);
    Check(runtime.allocations.size() == 2, "the written range was not reused");
    // gen_skip_ok = false always reads and compares.
    Check(BindSmallUniformReusing(runtime, memory, gen, false, 0, 0, 0, 0, 128),
          "valid small uniform rejected");
    Check(runtime.allocations.size() == 2, "an untrusted range allocated a copy");
    CheckBound(runtime, memory, 0, 128);
}

TEST_CASE("Uniform stream generation snapshots key per page and survive evictions", "[video_core]") {
    VideoCommon::GuestWriteGenerations gen;
    std::array<u64, 4> snap{};
    u32 size = 64;
    Check(gen.Snapshot(0x1000, size, snap) == 1, "one page range not keyable");
    Check(gen.Unchanged(0x1000, size, snap, 1), "range changed without a write");
    gen.NoteWrite(0x1FF0, 32); // ends on the next page
    Check(gen.Snapshot(0x1FF0, 32, snap) == 2, "two page range not keyable");
    Check(gen.Unchanged(0x1FF0, 32, snap, 2), "range changed without a write");
    gen.NoteWrite(0x1FF8, 4);
    Check(!gen.Unchanged(0x1FF0, 32, snap, 2), "write inside the range went unnoticed");
    Check(gen.Unchanged(0x2000, 0x1000, snap, 0) == false, "empty snapshot compared equal");
    // A range over four pages is not keyable (count 0) and never compares unchanged.
    Check(gen.Snapshot(0x3FF0, 0x4010, snap) == 0, "oversized range was keyed");
    Check(!gen.Unchanged(0x3FF0, 0x4010, snap, 0), "oversized range compared unchanged");
    // A write to a page evicted from the table gives it a fresh generation: a stale snapshot of
    // that page never compares unchanged again.
    Check(gen.Snapshot(0x1000, size, snap) == 1, "one page range not keyable");
    for (u64 page = 0; page < 4096 * 2; ++page) {
        gen.NoteWrite(page << 12, 16);
    }
    Check(!gen.Unchanged(0x1000, size, snap, 1), "an evicted page kept its old generation");
    Check(gen.Snapshot(0x1000, size, snap) == 1, "one page range not keyable");
    Check(gen.Unchanged(0x1000, size, snap, 1), "range changed without a write");
}

// The audit is the last test: it flips UniformStreamSkipBroken, which disables the skip for the
// rest of the process.
TEST_CASE("Uniform stream generation audit catches unnoticed writers", "[video_core]") {
    using VideoCommon::BindSmallUniformReusing;
    Memory memory;
    Runtime runtime;
    VideoCommon::GuestWriteGenerations gen;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    Check(!VideoCommon::UniformStreamSkipBroken(), "skip already broken at test start");
    // Mutate without bumping the generations: the skip reuses stale bytes until the sampled
    // audit (every 256th skip) reads the guest and disproves the proof.
    for (u32 index = 0; index < 2000 && !VideoCommon::UniformStreamSkipBroken(); ++index) {
        Check(BindSmallUniformReusing(runtime, memory, gen, true, 0, 0, 0, 0, 128),
              "valid small uniform rejected");
        memory.bytes[index % 128] = static_cast<u8>(index + 1);
    }
    Check(VideoCommon::UniformStreamSkipBroken(), "audit missed an unnoticed writer");
    // Once broken, every bind copies and keeps the bound data correct.
    const auto before = runtime.allocations.size();
    Check(BindSmallUniformReusing(runtime, memory, gen, true, 0, 0, 0, 0, 128),
          "valid small uniform rejected");
    Check(runtime.allocations.size() == before + 1, "broken skip still reused stale data");
    CheckBound(runtime, memory, 0, 128);
}

namespace {
// The production device memory manager exposes host pointers, so the bind compares the guest
// bytes in place; a manager that only offers ReadBlockUnsafe takes the copy-then-compare
// fallback the binds used before. Both must charge the same bind, which is what this measures.
// Neither carries the tests' bounds Check: Catch2's assertion bookkeeping would dwarf the copy.
struct CopyMemory {
    std::array<u8, 8192> bytes{};
    void ReadBlockUnsafe(u64 address, void* destination, u32 size) const {
        std::memcpy(destination, bytes.data() + address, size);
    }
};
struct SpanMemory {
    std::array<u8, 8192> bytes{};
    void ReadBlockUnsafe(u64 address, void* destination, u32 size) const {
        std::memcpy(destination, bytes.data() + address, size);
    }
    template <typename T>
    const T* GetPointer(u64 address) const {
        return reinterpret_cast<const T*>(bytes.data() + address);
    }
};
// Runtime without the tests' Catch2 bounds checks: their bookkeeping would dwarf the bind.
struct BenchRuntime {
    struct StreamUniformRef {
        size_t allocation;
    };
    std::vector<std::vector<u8>> allocations;
    StreamUniformRef last{}, bound{};
    u64 tick{1};
    u64 CurrentTick() const {
        return tick;
    }
    StreamUniformRef LastStreamUniformRef() const {
        return last;
    }
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        allocations.emplace_back(size);
        last = bound = StreamUniformRef{allocations.size() - 1};
        return allocations.back();
    }
    void RebindStreamUniform(StreamUniformRef ref, u32) {
        bound = ref;
    }
};
} // namespace

TEST_CASE("Uniform stream bind CPU benchmark", "[.][uniform_stream_bench]") {
    using VideoCommon::BindSmallUniformReusing;
    constexpr size_t binds = 2'000'000;
    constexpr size_t trials = 5;
    const auto measure = [&](auto& memory, u32 size) {
        BenchRuntime runtime;
        VideoCommon::GuestWriteGenerations gen;
        VideoCommon::ResetUniformStreamReuse<BenchRuntime>();
        REQUIRE(BindSmallUniformReusing(runtime, memory, gen, false, 0, 0, 0, 0, size));
        double best = std::numeric_limits<double>::max();
        for (size_t trial = 0; trial < trials; ++trial) {
            const auto start = std::chrono::steady_clock::now();
            for (size_t index = 0; index < binds; ++index) {
                BindSmallUniformReusing(runtime, memory, gen, false, 0, 0, 0, 0, size);
            }
            const auto elapsed = std::chrono::steady_clock::now() - start;
            best = (std::min)(best, std::chrono::duration<double, std::nano>(elapsed).count() /
                                        binds);
        }
        REQUIRE(runtime.allocations.size() == 1);
        return best;
    };
    CopyMemory copy_memory;   // the old copy-then-compare path
    SpanMemory span_memory;   // compare in place, as the device memory manager allows
    for (const u32 size : {64u, 256u, 1024u}) {
        const double copy_ns = measure(copy_memory, size);
        const double span_ns = measure(span_memory, size);
        std::cout << "Uniform stream bind size=" << size << ": copy_ns=" << copy_ns
                  << " inplace_ns=" << span_ns << " saved_ns=" << copy_ns - span_ns
                  << " (fastest of 5, CPU only, not gameplay FPS)\n";
    }
}
