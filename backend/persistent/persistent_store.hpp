#pragma once
// Phase 1.1: PersistentStore<T> — Generic mmap-backed fixed-record store
//
// Provides:
// - mmap/munmap lifecycle
// - Fixed-size record access by index
// - Grow via mremap (Linux) or remap fallback
// - Header management (count, capacity, next_id)

#include "persistent_records.hpp"
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

namespace brain19 {
namespace persistent {

template<typename RecordT>
class PersistentStore {
public:
    // initial_capacity = number of record slots
    PersistentStore(const std::string& filepath, const char magic[4], size_t initial_capacity = 4096)
        : filepath_(filepath), fd_(-1), mapped_(nullptr), mapped_size_(0)
    {
        std::memcpy(magic_, magic, 4);
        
        bool exists = (::access(filepath.c_str(), F_OK) == 0);
        
        fd_ = ::open(filepath.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("PersistentStore: cannot open " + filepath + ": " + strerror(errno));
        }
        
        if (!exists) {
            // New file: initialize with header + initial_capacity slots
            size_t file_size = sizeof(StoreHeader) + initial_capacity * sizeof(RecordT);
            if (::ftruncate(fd_, file_size) != 0) {
                ::close(fd_);
                throw std::runtime_error("PersistentStore: ftruncate failed");
            }
            
            map_file(file_size);
            
            // Write header
            auto* hdr = header();
            std::memcpy(hdr->magic, magic_, 4);
            hdr->version = 1;
            hdr->record_count = 0;
            hdr->capacity = initial_capacity;
            hdr->next_id = 1;
            
            ::msync(mapped_, sizeof(StoreHeader), MS_SYNC);
        } else {
            // Existing file: get size and map
            struct stat st;
            if (::fstat(fd_, &st) != 0) {
                ::close(fd_);
                throw std::runtime_error("PersistentStore: fstat failed");
            }
            
            if (static_cast<size_t>(st.st_size) < sizeof(StoreHeader)) {
                ::close(fd_);
                throw std::runtime_error("PersistentStore: file too small for header");
            }
            
            map_file(st.st_size);
            
            // Validate magic
            auto* hdr = header();
            if (std::memcmp(hdr->magic, magic_, 4) != 0) {
                unmap();
                ::close(fd_);
                throw std::runtime_error("PersistentStore: invalid magic in " + filepath);
            }
        }
    }
    
    ~PersistentStore() {
        if (mapped_) {
            sync();
            unmap();
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    
    // Non-copyable, movable
    PersistentStore(const PersistentStore&) = delete;
    PersistentStore& operator=(const PersistentStore&) = delete;
    
    // Access header
    StoreHeader* header() {
        return reinterpret_cast<StoreHeader*>(mapped_);
    }
    const StoreHeader* header() const {
        return reinterpret_cast<const StoreHeader*>(mapped_);
    }
    
    // Access record by slot index (0-based)
    RecordT* record(size_t index) {
        if (index >= header()->record_count) {
            throw std::out_of_range("PersistentStore: index out of range");
        }
        char* base = static_cast<char*>(mapped_) + sizeof(StoreHeader);
        return reinterpret_cast<RecordT*>(base + index * sizeof(RecordT));
    }
    
    const RecordT* record(size_t index) const {
        if (index >= header()->record_count) {
            throw std::out_of_range("PersistentStore: index out of range");
        }
        const char* base = static_cast<const char*>(mapped_) + sizeof(StoreHeader);
        return reinterpret_cast<const RecordT*>(base + index * sizeof(RecordT));
    }
    
    // Append a record, growing if needed. Returns slot index.
    size_t append(const RecordT& rec) {
        auto* hdr = header();
        if (hdr->record_count >= hdr->capacity) {
            grow(hdr->capacity * 2);
            hdr = header(); // re-acquire after remap
        }
        size_t slot = hdr->record_count;
        *record_at(slot) = rec;
        hdr->record_count++;
        return slot;
    }
    
    // Get/set next_id
    uint64_t next_id() const { return header()->next_id; }
    void set_next_id(uint64_t id) { header()->next_id = id; }
    
    // Record count
    uint64_t count() const { return header()->record_count; }
    
    // Sync to disk
    void sync() {
        if (mapped_ && mapped_size_ > 0) {
            ::msync(mapped_, mapped_size_, MS_SYNC);
        }
    }
    
    // Grow capacity to at least new_capacity slots
    void grow(size_t new_capacity) {
        auto* hdr = header();
        if (new_capacity <= hdr->capacity) return;
        
        size_t new_size = sizeof(StoreHeader) + new_capacity * sizeof(RecordT);
        
        // Extend file
        if (::ftruncate(fd_, new_size) != 0) {
            throw std::runtime_error("PersistentStore: grow ftruncate failed");
        }
        
#ifdef __linux__
        // Try mremap (Linux-specific, avoids unmap+mmap)
        void* new_map = ::mremap(mapped_, mapped_size_, new_size, MREMAP_MAYMOVE);
        if (new_map == MAP_FAILED) {
            throw std::runtime_error("PersistentStore: mremap failed");
        }
        mapped_ = new_map;
        mapped_size_ = new_size;
#else
        // Fallback: unmap + remap
        unmap();
        map_file(new_size);
#endif
        
        header()->capacity = new_capacity;
    }

private:
    // Internal: access by capacity (for append before record_count is incremented)
    RecordT* record_at(size_t index) {
        if (index >= header()->capacity) {
            throw std::out_of_range("PersistentStore: internal index out of range");
        }
        char* base = static_cast<char*>(mapped_) + sizeof(StoreHeader);
        return reinterpret_cast<RecordT*>(base + index * sizeof(RecordT));
    }

    void map_file(size_t size) {
        mapped_ = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped_ == MAP_FAILED) {
            mapped_ = nullptr;
            throw std::runtime_error("PersistentStore: mmap failed: " + std::string(strerror(errno)));
        }
        mapped_size_ = size;
    }
    
    void unmap() {
        if (mapped_ && mapped_size_ > 0) {
            ::munmap(mapped_, mapped_size_);
            mapped_ = nullptr;
            mapped_size_ = 0;
        }
    }
    
    std::string filepath_;
    char magic_[4];
    int fd_;
    void* mapped_;
    size_t mapped_size_;
};

} // namespace persistent
} // namespace brain19
