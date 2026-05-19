/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "params.h"
#include "utils.h"
#include "fwd_config.h"
#include SAGEATTN4_FWD_SPECIALIZATION_HEADER
#include "tile_scheduler.h"
#include "fwd_mainloop.cu"
#include "fwd_epilogue.cu"
#include "named_barrier.h"
#include "softmax_fused.h"

namespace flash::generated::SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE {

using namespace cute;

template <typename TileScheduler>
__global__ void __launch_bounds__(kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT Mainloop::Params const mainloop_params,
                    CUTE_GRID_CONSTANT Epilogue::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
                    ) {

    using CollectiveMainloop = Mainloop;
    using CollectiveEpilogue = Epilogue;
    static constexpr bool Is_causal = kIsCausal;

    using SoftType = ElementAccum;
    using ClusterShape = ClusterShape_MNK;

    static constexpr int NumMmaThreads = size(TiledMmaQK{});
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;

    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;


    enum class WarpGroupRole {
        Producer = 0,
        Consumer0 = 1,
        Consumer1 = 2
    };

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    bool const is_mainloop_producer_warp = warp_idx_in_warp_group == 0;

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index

    PipelineParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    pipeline_params_v.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_v.is_leader = warp_group_thread_idx == 0;
    pipeline_params_v.num_consumers = NumMmaThreads;

    PipelineParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_k.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_k.is_leader = warp_group_thread_idx == 0;
    pipeline_params_k.num_consumers = NumMmaThreads;

    PipelineParamsQ pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
    pipeline_params_q.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipelineQ::ThreadCategory::Producer
        : MainloopPipelineQ::ThreadCategory::Consumer;
    pipeline_params_q.is_leader = warp_group_thread_idx == 0;
    pipeline_params_q.num_consumers = NumMmaThreads;

    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q, ClusterShape{});
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params_v, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;
    __syncthreads();

    if (warp_group_role == WarpGroupRole::Producer) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        TileScheduler scheduler;

        PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
        PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

        for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params); work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, get<0>(block_coord));

            if (is_mainloop_producer_warp) {
                collective_mainloop.load_q(mainloop_params, scheduler_params,
                                           pipeline_q, smem_pipe_write_q,
                                           shared_storage, work_tile_info);
            }
            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopQLoaded));

            if (is_mainloop_producer_warp) {
                collective_mainloop.load_kv(mainloop_params, scheduler_params,
                                            pipeline_k, pipeline_v,
                                            smem_pipe_write_k, smem_pipe_write_v,
                                            shared_storage, work_tile_info);
            }
            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopMmaDone));

            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopOReady));
            if (is_mainloop_producer_warp && n_block_max > 0) {
                collective_epilogue.tma_store(shared_storage, epilogue_params, work_tile_info, scheduler_params);
            }
            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopEpilogueDone));
        }

        if (is_mainloop_producer_warp) {
            collective_mainloop.load_tail(pipeline_q, pipeline_k, pipeline_v, 
                                          smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v);
        }
    } else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
        cutlass::arch::warpgroup_reg_alloc<232>();
        TiledMmaPV tiled_mma_pv;
        TileScheduler scheduler{};
        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        PipelineStateQ smem_pipe_read_q;

        int work_idx = 0;

        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params); work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 2>(TileShape_MNK{}));
            // flash::Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax;
            flash::SoftmaxFused<2 * (2 * kBlockM / NumMmaThreads)> softmax_fused;
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);
            if (Is_causal && n_block_max <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
                cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopQLoaded));
                cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopMmaDone));
                collective_epilogue.store_zero(epilogue_params, threadIdx.x - NumCopyThreads, block_coord);
                cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopOReady));
                cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopEpilogueDone));
                continue;
            }

            collective_mainloop.mma(mainloop_params, pipeline_q, pipeline_k, pipeline_v,
                                    smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                                    tOrO, softmax_fused, n_block_max, threadIdx.x - NumCopyThreads, work_idx, m_block, shared_storage);
            collective_epilogue.mma_store(shared_storage, tiled_mma_pv, tOrO, threadIdx.x - NumCopyThreads); 
            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopOReady));
            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<uint32_t>(FP4NamedBarriers::MainloopEpilogueDone));
            ++work_idx;
        }
    }
}

void SAGEATTN4_FWD_RUN(Flash_fwd_params &params, cudaStream_t stream) {
    using Scheduler = flash::StaticPersistentTileScheduler;
    /*
    *    struct Arguments {
    Element const* ptr_Q;
    ShapeQKV const shape_Q;
    StrideQKV const stride_Q;
    Element const* ptr_K;
    ShapeQKV const shape_K;
    StrideQKV const stride_K;
    ShapeQKV const unpadded_shape_K;
    Element const* ptr_Vt;
    ShapeQKV const shape_Vt;
    StrideQKV const stride_Vt;
    ElementSF const* ptr_SFQ{nullptr};
    ShapeSF const shape_SFQ{};
    ElementSF const* ptr_SFK{nullptr};
    ShapeSF const shape_SFK{};
    ElementSF const* ptr_SFVt{nullptr};
    ShapeSF const shape_SFVt{};
    float const* ptr_ds;
    ShapeQKV const shape_ds;
    StrideQKV const stride_ds;

    float const* ptr_lambK;
    ShapeQKV const shape_lambK;
    StrideQKV const stride_lambK;

    float const softmax_scale_log2;
    };*/
    Mainloop::Params mainloop_params =
        Mainloop::to_underlying_arguments(Mainloop::Arguments{
            .ptr_Q = static_cast<Element const*>(params.q_ptr),
            .shape_Q = {params.seqlen_q, params.d, params.h, params.b},
            .stride_Q = {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},
            .ptr_K = static_cast<Element const*>(params.k_ptr),
            .shape_K = {params.seqlen_k, params.d, params.h_k, params.b},
            .stride_K = {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},
            .unpadded_shape_K = {params.unpadded_seqlen_k, params.d, params.h_k, params.b},
            .ptr_Vt = static_cast<Element const*>(params.v_ptr),
            .shape_Vt = {params.d, params.seqlen_k, params.h_k, params.b},
            .stride_Vt = {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},
            .ptr_SFQ = static_cast<ElementSF const*>(params.sfq_ptr),
            .shape_SFQ = {params.seqlen_q, params.d, params.h, params.b},
            .ptr_SFK = static_cast<ElementSF const*>(params.sfk_ptr),
            .shape_SFK = {params.seqlen_k, params.d, params.h_k, params.b},
            .ptr_SFVt = static_cast<ElementSF const*>(params.sfv_ptr),
            .shape_SFVt ={params.d, params.seqlen_k, params.h_k, params.b},
            .ptr_ds = static_cast<float const*>(params.delta_s_ptr),
            .shape_ds = {params.seqlen_s, params.seqlen_k, params.h_k, params.b},
            .stride_ds = {params.ds_row_stride, _1{}, params.ds_head_stride, params.ds_batch_stride},
            .ptr_lambK = static_cast<float const*>(params.lamb_k_ptr),
            .shape_lambK = {1, params.seqlen_k, params.h_k, params.b},
            .stride_lambK = {0, _1{}, params.lamb_k_head_stride, params.lamb_k_batch_stride},
            .unpadded_seqlen_q = params.unpadded_seqlen_q,
            .softmax_scale_log2 = params.scale_softmax_log2
        });
    Epilogue::Params epilogue_params =
        Epilogue::to_underlying_arguments({
            static_cast<ElementOut*>(params.o_ptr),
            {params.seqlen_q, params.d, params.h, params.b},
            {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},
            static_cast<float*>(params.softmax_lse_ptr),
            {_1{}, params.seqlen_q, params.h * params.seqlen_q},
        });

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, kBlockM);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape_MNK{})) * size<0>(ClusterShape_MNK{});
    Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b};
    Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    void *kernel = reinterpret_cast<void *>(compute_attn_ws<Scheduler>);
    int smem_size = sizeof(SharedStorage);
    if (smem_size >= 48 * 1024) {
       C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    static constexpr int ctaSize = kNWarps * 32;
    params.m_block_divmod = cutlass::FastDivmod(num_blocks_m);
    params.total_blocks = num_blocks_m * params.h * params.b;
    int const num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int const ctas_per_sm = smem_size <= 64 * 1024 ? 2 : 1;
    int const persistent_ctas = params.total_blocks < num_sm * ctas_per_sm
        ? params.total_blocks
        : num_sm * ctas_per_sm;
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, persistent_ctas);
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape_MNK{}), size<1>(ClusterShape_MNK{}), size<2>(ClusterShape_MNK{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace flash::generated::SAGEATTN4_FWD_SPECIALIZATION_NAMESPACE
