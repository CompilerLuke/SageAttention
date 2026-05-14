#pragma once

#include <cuda_runtime_api.h>

#include "params.h"

#define SAGEATTN4_FWD_SPECIALIZATION_HEADER "sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean1.h"
#define SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean1
#define SAGEATTN4_FWD_FUNCTION_PREFIX sageattn4_hdim128_bm128_bn128_s3_blockmean1_
#define SAGEATTN4_FWD_HEAD_DIM 128
#define SAGEATTN4_FWD_BLOCK_M 128
#define SAGEATTN4_FWD_BLOCK_N 128
#define SAGEATTN4_FWD_STAGES 3
#define SAGEATTN4_FWD_BLOCK_MEAN true
#define SAGEATTN4_FWD_IS_CAUSAL false

#define SAGEATTN4_JOIN_IMPL(a, b) a##b
#define SAGEATTN4_JOIN(a, b) SAGEATTN4_JOIN_IMPL(a, b)
#define SAGEATTN4_FWD_FUNCTION(name) SAGEATTN4_JOIN(name, fwd)
#define SAGEATTN4_FWD_RUN SAGEATTN4_FWD_FUNCTION(SAGEATTN4_FWD_FUNCTION_PREFIX)

namespace flash::generated::SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE {

void SAGEATTN4_FWD_RUN(Flash_fwd_params &params, cudaStream_t stream);

}  // namespace flash::generated::SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE
