#pragma once

#include <cuda_runtime_api.h>

#include "params.h"

#if defined(SAGEATTN4_FWD_BUILD_CAUSAL) && SAGEATTN4_FWD_BUILD_CAUSAL
#define SAGEATTN4_FWD_SPECIALIZATION_HEADER "sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean0_causal.h"
#define SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean0_causal
#define SAGEATTN4_FWD_FUNCTION_PREFIX sageattn4_hdim128_bm128_bn128_s3_blockmean0_causal_
#define SAGEATTN4_FWD_IS_CAUSAL true
#else
#define SAGEATTN4_FWD_SPECIALIZATION_HEADER "sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean0.h"
#define SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean0
#define SAGEATTN4_FWD_FUNCTION_PREFIX sageattn4_hdim128_bm128_bn128_s3_blockmean0_
#define SAGEATTN4_FWD_IS_CAUSAL false
#endif
#define SAGEATTN4_FWD_HEAD_DIM 128
#define SAGEATTN4_FWD_BLOCK_M 128
#define SAGEATTN4_FWD_BLOCK_N 128
#define SAGEATTN4_FWD_STAGES 3
#define SAGEATTN4_FWD_BLOCK_MEAN false

#define SAGEATTN4_JOIN_IMPL(a, b) a##b
#define SAGEATTN4_JOIN(a, b) SAGEATTN4_JOIN_IMPL(a, b)
#define SAGEATTN4_FWD_FUNCTION(name) SAGEATTN4_JOIN(name, fwd)
#define SAGEATTN4_FWD_RUN SAGEATTN4_FWD_FUNCTION(SAGEATTN4_FWD_FUNCTION_PREFIX)

namespace flash::generated::SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE {

void SAGEATTN4_FWD_RUN(Flash_fwd_params &params, cudaStream_t stream);

}  // namespace flash::generated::SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE
