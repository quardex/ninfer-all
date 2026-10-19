// The page bookkeeping of Qwen3.8-Flash-Next's shared KV pool (--kv-pool): which pool pages each
// sequence's block table row maps as it grows, that a reservation the pool cannot meet takes none,
// that a window caps what a sequence holds, and that released pages serve the next reservation,
// the lowest first. The oracle is the pages expected by hand for each step.
#include "models/qwen4_exp/kv_page_pool.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<std::int32_t> held_row(const KvPagePool& pool, std::uint32_t sequence) {
    const auto row = pool.table(sequence);
    return {row.begin(), row.begin() + std::ptrdiff_t(pool.held_pages(sequence))};
}

int run() {
    // Six pages, windows of four, three sequences.
    KvPagePool pool(6, 4, 3);
    require(pool.free_pages() == 6 && pool.held_pages(0) == 0, "a new pool holds every page free");
    require(pool.reserve(0, 2) && held_row(pool, 0) == std::vector<std::int32_t>{0, 1},
            "a sequence takes the lowest free pages first");
    require(pool.reserve(0, 2) && pool.held_pages(0) == 2 && pool.free_pages() == 4,
            "a reservation the sequence already holds takes nothing");
    require(pool.reserve(1, 4) && held_row(pool, 1) == std::vector<std::int32_t>{2, 3, 4, 5},
            "a second sequence maps the next pages");
    require(pool.free_pages() == 0, "the pool is spent");
    require(!pool.reserve(2, 1) && pool.held_pages(2) == 0,
            "a reservation the pool cannot meet fails");
    require(!pool.reserve(0, 3) && held_row(pool, 0) == std::vector<std::int32_t>{0, 1} &&
                pool.free_pages() == 0,
            "a failed reservation takes none of the pages it could have");
    pool.release(1);
    require(pool.free_pages() == 4 && pool.held_pages(1) == 0, "a release returns every page");
    require(pool.reserve(2, 3) && held_row(pool, 2) == std::vector<std::int32_t>{2, 3, 4},
            "released pages serve the next reservation, the lowest first");
    require(!pool.reserve(0, 4) && pool.held_pages(0) == 2,
            "growing past the free pages fails and keeps what the sequence held");
    pool.release(2);
    require(pool.reserve(0, 9) && held_row(pool, 0) == std::vector<std::int32_t>{0, 1, 2, 3},
            "a sequence grows in place, capped at its window");
    require(pool.free_pages() == 2 && pool.window_pages() == 4 && pool.pool_pages() == 6,
            "the pool keeps its geometry");
    // The row keeps stale entries past what a sequence holds; only held_pages() of it count.
    pool.release(0);
    require(pool.reserve(1, 1) && held_row(pool, 1) == std::vector<std::int32_t>{0},
            "a sequence released after growing frees its pages for any other");
    bool invalid = false;
    try {
        KvPagePool none(0, 4, 3);
    } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid, "a pool without pages is refused");
    bool outside = false;
    try {
        (void)pool.table(3);
    } catch (const std::out_of_range&) { outside = true; }
    require(outside, "a sequence past the pool's is refused");
    return 0;
}

} // namespace

int main() {
    try {
        const int result = run();
        std::cout << "PASS qwen4_exp kv page pool\n";
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
