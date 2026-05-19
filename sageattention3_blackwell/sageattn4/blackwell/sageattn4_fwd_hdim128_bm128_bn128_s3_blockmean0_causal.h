/*
 * Concrete SageAttention4 Blackwell forward specialization.
 *
 * Spec:
 *   head_dim=128, block_m=128, block_n=128, stages=3, cluster_m=1
 *   block_mean=false, is_causal=true, element_pair=nv_float4<float_e2m1>, output=bfloat16
 *   hand-maintained concrete aliases for IDEs and CUDA compilation.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cute/atom/copy_traits.hpp>
namespace cute {
template <class... Args>
struct Copy_Atom;
}
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/pipeline/pipeline.hpp>

#include "cute_extension.h"
#include "params.h"
#include "shared_storage.h"

namespace flash::generated::sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean0_causal {
    using namespace cute;

    static constexpr int kHeadDim = 128;
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kStages = 3;
    static constexpr int kClusterM = 1;
    static constexpr bool kBlockMean = false;
    static constexpr bool kIsCausal = true;
    static constexpr int kNWarps = 12;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kNumSFQK = kHeadDim / 16;
    static constexpr int kNumSFPV = kBlockN / 16;
    static constexpr int kSFVectorSize = 16;

    static constexpr int QUANT_BLOCK_SIZE = 16;

    using M = Int<128>;
    using N = Int<128>;
    using K = Int<128>;
    using Stage = Int<3>;
    using ElementPairType = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using OutputType = cutlass::bfloat16_t;
    using ElementSF = cutlass::float_ue4m3_t;
    using Element = cutlass::float_e2m1_t;
    using ElementAccum = float;
    using ElementOut = cutlass::bfloat16_t;
    using index_t = int64_t;
    using TileShape_MNK = Shape<M, N, K>;
    using ClusterShape_MNK = Shape<Int<1>, Int<1>, Int<1>>;
    using PermTileM = M;
    using PermTileN = Int<32>;
    using PermTileK = K;
    using AtomLayoutMNK = decltype(
        make_layout(
            make_shape(Int<8>{}, Int<1>{}, Int<1>{}),
            make_stride(Int<1>{}, Int<0>{}, Int<0>{})
        )
    );
    using ElementQMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<cutlass::float_e2m1_t>());
    using ElementKMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<cutlass::float_e2m1_t>());
    using TiledMmaQK = decltype(make_tiled_mma(
        SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        decltype(make_layout(
            make_shape(Int<8>{}, Int<1>{}, Int<1>{}),
            make_stride(Int<1>{}, Int<0>{}, Int<0>{})
        )){},
        Tile<Int<128>, Int<32>, Int<128>>{}
    ));
    using TiledMmaPV = decltype(make_tiled_mma(
        SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        decltype(make_layout(
            make_shape(Int<8>{}, Int<1>{}, Int<1>{}),
            make_stride(Int<1>{}, Int<0>{}, Int<0>{})
        )){},
        Tile<Int<128>, Int<32>, Int<128>>{}
    ));
    using GmemTiledCopy = SM90_TMA_LOAD;
    using GmemTiledCopySF = SM90_TMA_LOAD;
    using SmemLayoutAtomQ = UMMA::Layout_K_SW64_Atom<cutlass::float_e2m1_t>;
    using SmemLayoutAtomK = UMMA::Layout_K_SW64_Atom<cutlass::float_e2m1_t>;
    using SmemLayoutAtomV = UMMA::Layout_K_SW64_Atom<cutlass::float_e2m1_t>;
    using SmemLayoutAtomVt = UMMA::Layout_K_SW64_Atom<cutlass::float_e2m1_t>;
    using SmemLayoutQ = decltype(
        make_composed_layout(
            Swizzle<2, 4, 3>{},
            smem_ptr_flag_bits<4>{},
            make_layout(
                make_shape(make_shape(Int<8>{}, Int<16>{}), make_shape(Int<128>{}, Int<1>{})),
                make_stride(make_stride(Int<128>{}, Int<1024>{}), make_stride(Int<1>{}, Int<0>{}))
            )
        )
    );
    using SmemLayoutK = decltype(
        make_composed_layout(
            Swizzle<2, 4, 3>{},
            smem_ptr_flag_bits<4>{},
            make_layout(
                make_shape(make_shape(Int<8>{}, Int<16>{}), make_shape(Int<128>{}, Int<1>{}), make_shape(Int<1>{}, Int<3>{})),
                make_stride(make_stride(Int<128>{}, Int<1024>{}), make_stride(Int<1>{}, Int<0>{}), make_stride(Int<0>{}, Int<16384>{}))
            )
        )
    );
    using SmemLayoutV = decltype(
        make_composed_layout(
            Swizzle<2, 4, 3>{},
            smem_ptr_flag_bits<4>{},
            make_layout(
                make_shape(make_shape(Int<8>{}, Int<16>{}), make_shape(Int<128>{}, Int<1>{}), make_shape(Int<1>{}, Int<3>{})),
                make_stride(make_stride(Int<128>{}, Int<1024>{}), make_stride(Int<1>{}, Int<0>{}), make_stride(Int<0>{}, Int<16384>{}))
            )
        )
    );
    using SmemLayoutVt = decltype(
        make_composed_layout(
            Swizzle<2, 4, 3>{},
            smem_ptr_flag_bits<4>{},
            make_layout(
                make_shape(make_shape(Int<8>{}, Int<16>{}), make_shape(Int<128>{}, Int<1>{}), make_shape(Int<1>{}, Int<3>{})),
                make_stride(make_stride(Int<128>{}, Int<1024>{}), make_stride(Int<1>{}, Int<0>{}), make_stride(Int<0>{}, Int<16384>{}))
            )
        )
    );
    using SmemLayoutAtomDS = decltype(
        make_layout(
            make_shape(Int<128>{}, Int<128>{}),
            make_stride(Int<0>{}, Int<1>{})
        )
    );
    using SmemLayoutDS = decltype(
        make_layout(
            make_shape(make_shape(Int<128>{}, Int<1>{}), make_shape(Int<128>{}, Int<1>{}), make_shape(Int<1>{}, Int<3>{})),
            make_stride(make_stride(Int<0>{}, Int<0>{}), make_stride(Int<1>{}, Int<0>{}), make_stride(Int<0>{}, Int<128>{}))
        )
    );

    using SmemLayoutAtomLambK = decltype(
    make_layout(
        make_shape(Int<128>{}, Int<128>{}),
        make_stride(Int<0>{}, Int<1>{})
    )
    );
    using SmemLayoutLambK = decltype(
        make_layout(
            make_shape(make_shape(Int<128>{}, Int<1>{}), make_shape(Int<128>{}, Int<1>{}), make_shape(Int<1>{}, Int<3>{})),
            make_stride(make_stride(Int<0>{}, Int<0>{}), make_stride(Int<1>{}, Int<0>{}), make_stride(Int<0>{}, Int<128>{}))
        )
    );

    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::float_e2m1_t>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::float_e2m1_t>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<cutlass::float_ue4m3_t>, cutlass::float_ue4m3_t>;
    using SmemCopyAtomDS = Copy_Atom<UniversalCopy<float>, float>;
    using SmemCopyAtomLamb = Copy_Atom<UniversalCopy<float>, float>;
    using SfAtom = decltype(
        make_layout(
            make_shape(make_shape(Int<16>{}, Int<4>{}), make_shape(Int<16>{}, Int<4>{})),
            make_stride(make_stride(Int<16>{}, Int<4>{}), make_stride(Int<0>{}, Int<1>{}))
        )
    );
    using LayoutSF = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), int32_t{}), make_shape(make_shape(Int<16>{}, Int<4>{}), int32_t{}), make_shape(Int<1>{}, int32_t{}), make_shape(Int<1>{}, int32_t{})),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), int32_t{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<256>{}), make_stride(Int<0>{}, int32_t{}), make_stride(Int<0>{}, int32_t{}))
        )
    );
    using SmemLayoutAtomSFQ = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{})),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}))
        )
    );
    using SmemLayoutAtomSFK = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{})),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}))
        )
    );
    using SmemLayoutAtomSFV = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{})),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}))
        )
    );
    using SmemLayoutAtomSFVt = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{})),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}))
        )
    );
    using LayoutSFP = decltype(
        make_layout(
            make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{}),
            make_stride(make_stride(Int<0>{}, Int<1>{}), Int<0>{}, Int<4>{})
        )
    );
    using LayoutP = decltype(
        make_layout(
            make_shape(make_shape(Int<8>{}, Int<2>{}, Int<2>{}), Int<1>{}, Int<2>{}),
            make_stride(make_stride(Int<1>{}, Int<8>{}, Int<16>{}), Int<0>{}, Int<32>{})
        )
    );
    using SmemLayoutSFQ = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{})),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}))
        )
    );

    using SmemLayoutSFK = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{}), Int<3>{}),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}), Int<1024>{})
        )
    );
    using SmemLayoutSFV = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{}), Int<3>{}),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}), Int<1024>{})
        )
    );
    using SmemLayoutSFVt = decltype(
        make_layout(
            make_shape(make_shape(make_shape(Int<16>{}, Int<4>{}), Int<2>{}), make_shape(make_shape(Int<16>{}, Int<4>{}), Int<1>{}, Int<2>{}), Int<3>{}),
            make_stride(make_stride(make_stride(Int<16>{}, Int<4>{}), Int<256>{}), make_stride(make_stride(Int<0>{}, Int<1>{}), Int<4>{}, Int<512>{}), Int<1024>{})
        )
    );
    using LayoutDS = decltype(
        make_layout(
            make_shape(make_shape(Int<128>{}, int32_t{}), make_shape(Int<128>{}, int32_t{}), make_shape(Int<1>{}, int32_t{}), make_shape(Int<1>{}, int32_t{})),
            make_stride(make_stride(Int<0>{}, int32_t{}), make_stride(Int<1>{}, Int<128>{}), make_stride(Int<0>{}, int32_t{}), make_stride(Int<0>{}, int32_t{}))
        )
    );
    using LayoutLambK = decltype(
        make_layout(
            make_shape(make_shape(Int<128>{}, int32_t{}), make_shape(Int<128>{}, int32_t{}), make_shape(Int<1>{}, int32_t{}), make_shape(Int<1>{}, int32_t{})),
            make_stride(make_stride(Int<0>{}, int32_t{}), make_stride(Int<1>{}, Int<128>{}), make_stride(Int<0>{}, int32_t{}), make_stride(Int<0>{}, int32_t{}))
        )
    );

    using SmemLayoutAtomO = GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;
    using SmemLayoutO = decltype(
        make_composed_layout(
            Swizzle<3, 4, 3>{},
            smem_ptr_flag_bits<16>{},
            make_layout(
                make_shape(make_shape(Int<8>{}, Int<16>{}), make_shape(Int<64>{}, Int<2>{})),
                make_stride(make_stride(Int<64>{}, Int<512>{}), make_stride(Int<1>{}, Int<8192>{}))
            )
        )
    );
    using SharedStorage = ::SharedStorageQKVOwithSF<
        kStages,
        cutlass::float_e2m1_t,
        cutlass::float_ue4m3_t,
        cutlass::bfloat16_t,
        SmemLayoutQ,
        SmemLayoutK,
        SmemLayoutV,
        SmemLayoutDS,
        SmemLayoutO,
        SmemLayoutSFQ,
        SmemLayoutSFK,
        SmemLayoutSFVt,
        SmemLayoutLambK>;
    using MainloopPipeline = cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = cutlass::PipelineState<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = MainloopPipelineQ::Params;
    using PipelineStateQ = cutlass::PipelineState<1>;

    template <class ProblemShape>
    CUTE_HOST_DEVICE constexpr auto
    tile_atom_to_shape_SFQKV(ProblemShape problem_shape) {
      auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(Seqlen, Dim, HeadNum, Batch), Step<_2,_1,_3,_4>{});
    }

    template <class ProblemShape>
    CUTE_HOST_DEVICE constexpr auto
    tile_atom_to_shape_SFVt(ProblemShape problem_shape) {
      auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(Dim, Seqlen, HeadNum, Batch), Step<_2,_1,_3,_4>{});
    }

    static constexpr int kMmaNSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / kSFVectorSize;

}  // namespace flash::generated::sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean0_causal
