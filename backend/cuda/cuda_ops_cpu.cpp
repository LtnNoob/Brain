// cuda/cuda_ops_cpu.cpp — CPU fallback implementation
// Compiled when USE_CUDA is NOT defined
#include "cuda_ops.h"

namespace brain19 {
namespace cuda {

bool gpu_available() { return false; }
size_t gpu_vram_bytes() { return 0; }

bool ridge_solve(const RidgeParams& /*params*/, double* /*w_out*/) {
    return false;  // signal caller to use its own CPU path
}

bool matmul_AtB(const double* /*A*/, const double* /*B*/, double* /*C*/,
                size_t /*M*/, size_t /*K*/, size_t /*N*/) {
    return false;
}

bool matrix_invert(const double* /*mat*/, double* /*inv_out*/, size_t /*D*/) {
    return false;
}

} // namespace cuda
} // namespace brain19
