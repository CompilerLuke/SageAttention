#pragma once

#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "named_barrier.h"

using namespace cute;

template <
    int kStages,
    int EpiStages,
    typename Element,
    typename ElementSF,
    typename OutputType,
    typename SmemLayoutQ,
    typename SmemLayoutK,
    typename SmemLayoutV,
    typename SmemLayoutDS,
    typename SmemLayoutO,
    typename SmemLayoutSFQ,
    typename SmemLayoutSFK,
    typename SmemLayoutSFV,
    typename SmemLayoutLambK
>
struct SharedStorageQKVOwithSF : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    ArrayEngine<ElementSF, cosize_v<SmemLayoutSFQ>> smem_SFQ;
    ArrayEngine<ElementSF, cosize_v<SmemLayoutSFK>> smem_SFK;
    ArrayEngine<ElementSF, cosize_v<SmemLayoutSFV>> smem_SFV;
    ArrayEngine<ElementSF, cosize_v<SmemLayoutLambK>> smem_lamb_K;
    alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutDS>> smem_ds;
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;

    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        alignas(16) typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o;
        int tile_count_semaphore;
    };
};
