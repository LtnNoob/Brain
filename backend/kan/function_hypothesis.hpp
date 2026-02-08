#pragma once

#include <cstddef>
#include <chrono>
#include <memory>

namespace brain19 {

class KANModule;

// FunctionHypothesis: Pure data wrapper
// Contains NO logic, only metadata about learned function
struct FunctionHypothesis {
    // Dimensions
    size_t input_dim;
    size_t output_dim;
    
    // Reference to trained module
    std::shared_ptr<KANModule> module;
    
    // Training metadata
    size_t training_iterations;
    double training_error;
    std::chrono::system_clock::time_point created_at;
    
    // Default constructor
    FunctionHypothesis()
        : input_dim(0)
        , output_dim(0)
        , module(nullptr)
        , training_iterations(0)
        , training_error(0.0)
        , created_at(std::chrono::system_clock::now())
    {}
    
    // Full constructor
    FunctionHypothesis(
        size_t in_dim,
        size_t out_dim,
        std::shared_ptr<KANModule> mod,
        size_t iterations,
        double error
    )
        : input_dim(in_dim)
        , output_dim(out_dim)
        , module(mod)
        , training_iterations(iterations)
        , training_error(error)
        , created_at(std::chrono::system_clock::now())
    {}
    
    // Validation
    bool is_valid() const {
        return input_dim > 0 && output_dim > 0 && module != nullptr;
    }
};

} // namespace brain19
