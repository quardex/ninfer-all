#pragma once

// The page bookkeeping of a KV pool that Qwen3.8-Flash-Next's sequences share
// (ExecutorOptions::kv_pool_tokens): which of the pool's pages each sequence's block table maps.

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen4_exp {

// `pool_pages` pages of KV planes for `sequences` sequences, each mapping at most `window_pages`
// pages (its window) through its row of the block table. A sequence takes pages as its positions
// grow, the lowest free ones first, and gives all of them back at release(); a reservation the
// pool cannot meet takes none.
class KvPagePool {
public:
    KvPagePool() = default;
    KvPagePool(std::uint32_t pool_pages, std::uint32_t window_pages, std::uint32_t sequences)
        : pool_pages_(pool_pages), window_pages_(window_pages), held_(sequences),
          table_(std::size_t(window_pages) * sequences, 0) {
        if (pool_pages == 0 || window_pages == 0 || sequences == 0) {
            throw std::invalid_argument("KvPagePool: needs pages, a window and sequences");
        }
        free_.reserve(pool_pages);
        // Taken from the back: the lowest pages first.
        for (std::uint32_t p = 0; p < pool_pages; ++p) {
            free_.push_back(static_cast<std::int32_t>(pool_pages - 1 - p));
        }
    }

    // Makes `sequence` hold its window's first `pages` pages (at most the window): true when it
    // holds them now, the ones it took entered at its table row's [held before, pages); false,
    // taking none, when the pool lacks some.
    bool reserve(std::uint32_t sequence, std::uint32_t pages) {
        auto& held = held_.at(sequence);
        if (pages > window_pages_) { pages = window_pages_; }
        if (pages <= held.size()) { return true; }
        const std::size_t more = pages - held.size();
        if (free_.size() < more) { return false; }
        std::int32_t* const row = table_.data() + std::size_t(sequence) * window_pages_;
        for (std::size_t k = 0; k < more; ++k) {
            row[held.size()] = free_.back();
            held.push_back(free_.back());
            free_.pop_back();
        }
        return true;
    }

    // Gives the sequence's pages back; its table row keeps stale entries past what it holds.
    void release(std::uint32_t sequence) {
        auto& held = held_.at(sequence);
        free_.insert(free_.end(), held.rbegin(), held.rend());
        held.clear();
    }

    [[nodiscard]] std::uint32_t pool_pages() const noexcept { return pool_pages_; }
    [[nodiscard]] std::uint32_t window_pages() const noexcept { return window_pages_; }
    [[nodiscard]] std::size_t free_pages() const noexcept { return free_.size(); }
    [[nodiscard]] std::size_t held_pages(std::uint32_t sequence) const {
        return held_.at(sequence).size();
    }
    // The sequence's block table row (window_pages entries; the first held_pages() are its pages).
    [[nodiscard]] std::span<const std::int32_t> table(std::uint32_t sequence) const {
        if (sequence >= held_.size()) { throw std::out_of_range("KvPagePool: no such sequence"); }
        return {table_.data() + std::size_t(sequence) * window_pages_, window_pages_};
    }

private:
    std::uint32_t pool_pages_   = 0;
    std::uint32_t window_pages_ = 0;
    std::vector<std::int32_t> free_;
    std::vector<std::vector<std::int32_t>> held_;
    std::vector<std::int32_t> table_; // [sequences][window_pages]
};

} // namespace ninfer::models::qwen4_exp
