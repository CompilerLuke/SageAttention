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
    struct QSmemStorage {
        alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        ArrayEngine<ElementSF, cosize_v<SmemLayoutSFQ>> smem_SFQ;
    };

    struct KVSmemStorage {
        alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        ArrayEngine<ElementSF, cosize_v<SmemLayoutSFK>> smem_SFK;
        ArrayEngine<ElementSF, cosize_v<SmemLayoutSFV>> smem_SFV;
        alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutLambK>> smem_lambK;
        alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutDS>> smem_ds;
    };

    union MainloopSmemStorage {
        QSmemStorage q_smem;
        KVSmemStorage kv_smem;
    };

    union {
        MainloopSmemStorage mainloop_smem;
        alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };

    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        alignas(16) typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o;
        int tile_count_semaphore;
    };
};
