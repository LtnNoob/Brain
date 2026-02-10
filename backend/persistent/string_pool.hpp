#pragma once
// Phase 1.1: StringPool — mmap-backed append-only string storage
//
// Strings are stored contiguously after a 64-byte header.
// Each string is referenced by (offset, length) from concept/relation records.
// No null terminators stored — length is explicit.

#include "persistent_records.hpp"
#include <string>
#include <string_view>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace brain19 {
namespace persistent {

class StringPool {
public:
    // initial_size = total file size including header
    StringPool(const std::string& filepath, size_t initial_size = 1024 * 1024)
        : filepath_(filepath), fd_(-1), mapped_(nullptr), mapped_size_(0)
    {
        bool exists = (::access(filepath.c_str(), F_OK) == 0);
        
        fd_ = ::open(filepath.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("StringPool: cannot open " + filepath);
        }
        
        if (!exists) {
            if (initial_size < sizeof(StringPoolHeader)) {
                initial_size = sizeof(StringPoolHeader) + 1024 * 1024;
            }
            if (::ftruncate(fd_, initial_size) != 0) {
                ::close(fd_);
                throw std::runtime_error("StringPool: ftruncate failed");
            }
            map_file(initial_size);
            
            auto* hdr = header();
            std::memcpy(hdr->magic, "B19S", 4);
            hdr->version = 1;
            hdr->used_bytes = 0;  // offset relative to data start (after header)
            hdr->capacity = initial_size;
            ::msync(mapped_, sizeof(StringPoolHeader), MS_SYNC);
        } else {
            struct stat st;
            ::fstat(fd_, &st);
            map_file(st.st_size);
            
            auto* hdr = header();
            if (std::memcmp(hdr->magic, "B19S", 4) != 0) {
                unmap();
                ::close(fd_);
                throw std::runtime_error("StringPool: invalid magic");
            }
        }
    }
    
    ~StringPool() {
        if (mapped_) {
            sync();
            unmap();
        }
        if (fd_ >= 0) ::close(fd_);
    }
    
    StringPool(const StringPool&) = delete;
    StringPool& operator=(const StringPool&) = delete;
    
    // Append string, returns (offset, length). Offset is relative to data start.
    std::pair<uint32_t, uint32_t> append(const std::string& str) {
        auto* hdr = header();
        size_t needed = hdr->used_bytes + str.size() + sizeof(StringPoolHeader);
        
        if (needed > hdr->capacity) {
            grow(needed * 2);
            hdr = header();
        }
        
        if (hdr->used_bytes + str.size() > UINT32_MAX) {
            throw std::overflow_error("StringPool: exceeded 4GB offset limit");
        }
        uint32_t offset = static_cast<uint32_t>(hdr->used_bytes);
        uint32_t length = static_cast<uint32_t>(str.size());
        
        char* dest = data_base() + offset;
        std::memcpy(dest, str.data(), str.size());
        
        hdr->used_bytes += str.size();
        return {offset, length};
    }
    
    // Retrieve string by offset+length
    std::string get(uint32_t offset, uint32_t length) const {
        if (offset + length > header()->used_bytes) {
            throw std::out_of_range("StringPool: read beyond used bytes");
        }
        return std::string(data_base() + offset, length);
    }
    
    void sync() {
        if (mapped_) ::msync(mapped_, mapped_size_, MS_SYNC);
    }

private:
    StringPoolHeader* header() {
        return reinterpret_cast<StringPoolHeader*>(mapped_);
    }
    const StringPoolHeader* header() const {
        return reinterpret_cast<const StringPoolHeader*>(mapped_);
    }
    
    char* data_base() {
        return static_cast<char*>(mapped_) + sizeof(StringPoolHeader);
    }
    const char* data_base() const {
        return static_cast<const char*>(mapped_) + sizeof(StringPoolHeader);
    }
    
    void grow(size_t new_size) {
        if (::ftruncate(fd_, new_size) != 0) {
            throw std::runtime_error("StringPool: grow failed");
        }
#ifdef __linux__
        void* new_map = ::mremap(mapped_, mapped_size_, new_size, MREMAP_MAYMOVE);
        if (new_map == MAP_FAILED) throw std::runtime_error("StringPool: mremap failed");
        mapped_ = new_map;
        mapped_size_ = new_size;
#else
        unmap();
        map_file(new_size);
#endif
        header()->capacity = new_size;
    }
    
    void map_file(size_t size) {
        mapped_ = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; throw std::runtime_error("StringPool: mmap failed"); }
        mapped_size_ = size;
    }
    
    void unmap() {
        if (mapped_) { ::munmap(mapped_, mapped_size_); mapped_ = nullptr; mapped_size_ = 0; }
    }
    
    std::string filepath_;
    int fd_;
    void* mapped_;
    size_t mapped_size_;
};

} // namespace persistent
} // namespace brain19
