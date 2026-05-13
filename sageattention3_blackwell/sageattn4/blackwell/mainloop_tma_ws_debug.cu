#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/atom/copy_traits.hpp>
namespace cute {
template <class... Args>
struct Copy_Atom;
}
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm90.hpp>

#include "kernel_traits.h"
#include "mainloop_tma_ws.h"

#ifndef SAGEATTN4_DEBUG_HEADDIM
#define SAGEATTN4_DEBUG_HEADDIM 128
#endif

#ifndef SAGEATTN4_DEBUG_BLOCK_M
#define SAGEATTN4_DEBUG_BLOCK_M 128
#endif

#ifndef SAGEATTN4_DEBUG_BLOCK_N
#define SAGEATTN4_DEBUG_BLOCK_N 128
#endif

#ifndef SAGEATTN4_DEBUG_STAGES
#define SAGEATTN4_DEBUG_STAGES 3
#endif

#ifndef SAGEATTN4_DEBUG_CLUSTER_M
#define SAGEATTN4_DEBUG_CLUSTER_M 1
#endif

#ifndef SAGEATTN4_DEBUG_BLOCK_MEAN
#define SAGEATTN4_DEBUG_BLOCK_MEAN 1
#endif

#ifndef SAGEATTN4_DEBUG_IS_CAUSAL
#define SAGEATTN4_DEBUG_IS_CAUSAL 0
#endif

#ifndef SAGEATTN4_DEBUG_THREAD_IDX
#define SAGEATTN4_DEBUG_THREAD_IDX 0
#endif

namespace {

template <class T>
std::string type_name() {
#if defined(__GNUG__)
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> demangled{
      abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
      std::free};
  if (status == 0 && demangled) {
    return demangled.get();
  }
#endif
  return typeid(T).name();
}

template <class T>
void print_type(char const* label) {
  std::cout << "  " << label << ": " << type_name<T>() << "\n";
}

template <class T>
void print_cute(char const* label, T const& value) {
  std::printf("  %-32s ", label);
  cute::print(value);
  std::printf("\n");
}

template <class Layout>
void dump_layout(char const* label) {
  std::printf("%s\n", label);
  print_cute("layout", Layout{});
  print_cute("shape", cute::shape(Layout{}));
  print_cute("size", cute::size(Layout{}));
  print_cute("cosize", cute::cosize(Layout{}));
}

template <class Tensor>
void dump_tensor(char const* label, Tensor const& tensor) {
  std::printf("%s\n", label);
  print_cute("tensor", tensor);
  print_cute("layout", tensor.layout());
  print_cute("shape", cute::shape(tensor));
  print_cute("size", cute::size(tensor));
}

template <class Coord>
int coord_m(Coord const& coord) {
  return int(cute::get<0, 0>(coord)) * 16 + int(cute::get<0, 1>(coord));
}

template <class Coord>
int coord_n_or_k(Coord const& coord) {
  return int(cute::get<1, 0>(coord));
}

template <class Coord>
int coord_stage(Coord const& coord) {
  if constexpr (cute::rank(Coord{}) > 2) {
    return int(cute::get<2, 1>(coord));
  } else {
    return 0;
  }
}

template <class Tensor>
void dump_access_map_csv(char const* label, int thread_idx, Tensor const& coord_tensor) {
  for (int i = 0; i < int(cute::size(coord_tensor)); ++i) {
    auto coord = coord_tensor(i);
    std::cout << label << "," << thread_idx << "," << i << ","
              << coord_m(coord) << "," << coord_n_or_k(coord) << ","
              << coord_stage(coord) << "\n";
  }
}

template <class Layout>
auto debug_convert_to_conversion_layout(Layout mma_layout) {
  static_assert(cute::rank(mma_layout) == 3, "MMA layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(cute::rank(cute::get<0>(cute::shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

  constexpr int MmaAtomN = cute::size<0, 0>(mma_layout);
  constexpr int MmaAtomM = cute::size<0, 1>(mma_layout);
  constexpr int MmaN = cute::size<2>(mma_layout);

  static_assert(MmaAtomN == 8, "MmaAtomN should be 8.");
  static_assert(MmaAtomM == 2, "MmaAtomM should be 2.");
  static_assert(MmaN % 2 == 0, "MmaN should be a multiple of 2.");

  auto mma_n_division = cute::zipped_divide(
      cute::layout<2>(mma_layout), cute::make_tile(cute::_2{}));
  return cute::make_layout(
      cute::make_layout(cute::layout<0, 0>(mma_layout),
                        cute::make_layout(cute::layout<0, 1>(mma_layout),
                                          cute::layout<0>(mma_n_division))),
      cute::layout<1>(mma_layout),
      cute::layout<1>(mma_n_division));
}

template <class Traits, bool IsCausal>
void dump_mainloop_traits(bool emit_access_map_csv) {
  using Mainloop = flash::CollectiveMainloopFwd<Traits, IsCausal>;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using TileShape_MNK = typename Traits::TileShape_MNK;

  if (emit_access_map_csv) {
    typename Traits::TiledMmaQK tiled_mma_qk;
    typename Traits::TiledMmaPV tiled_mma_pv;

    auto sQ = cute::make_tensor(cute::make_smem_ptr<Element>(nullptr), typename Traits::SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr<Element>(nullptr), typename Traits::SmemLayoutK{});
    auto sVt = cute::make_tensor(cute::make_smem_ptr<Element>(nullptr), typename Traits::SmemLayoutVt{});

    auto smem_tiled_copy_Q = cute::make_tiled_copy_A(typename Traits::SmemCopyAtomQ{}, tiled_mma_qk);
    auto smem_tiled_copy_K = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomKV{}, tiled_mma_qk);
    auto smem_tiled_copy_V = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomKV{}, tiled_mma_pv);

    auto cQ = cute::make_identity_tensor(cute::shape(sQ));
    auto cK = cute::make_identity_tensor(cute::shape(sK));
    auto cVt = cute::make_identity_tensor(cute::shape(sVt));

    std::cout << "layout,thread,value,m,n_or_k,stage\n";
    for (int thread_idx = 0; thread_idx < 256; ++thread_idx) {
      auto q_thread = smem_tiled_copy_Q.get_thread_slice(thread_idx);
      auto k_thread = smem_tiled_copy_K.get_thread_slice(thread_idx);
      auto v_thread = smem_tiled_copy_V.get_thread_slice(thread_idx);
      dump_access_map_csv("SmemLayoutQ", thread_idx, q_thread.partition_S(cQ));
      dump_access_map_csv("SmemLayoutK", thread_idx, k_thread.partition_S(cK));
      dump_access_map_csv("SmemLayoutVt", thread_idx, v_thread.partition_S(cVt));
    }
    return;
  }

  std::cout << "SageAttention4 mainloop_tma_ws debug\n";
  std::cout << "Config\n";
  std::cout << "  kHeadDim: " << Traits::kHeadDim << "\n";
  std::cout << "  kBlockM: " << Traits::kBlockM << "\n";
  std::cout << "  kBlockN: " << Traits::kBlockN << "\n";
  std::cout << "  kStages: " << Traits::kStages << "\n";
  std::cout << "  EpiStages: " << Traits::EpiStages << "\n";
  std::cout << "  kNWarps: " << Traits::kNWarps << "\n";
  std::cout << "  kNThreads: " << Traits::kNThreads << "\n";
  std::cout << "  kClusterM: " << Traits::kClusterM << "\n";
  std::cout << "  BlockMean: " << Traits::BlockMean << "\n";
  std::cout << "  IsCausal: " << IsCausal << "\n";
  std::cout << "  Debug consumer thread_idx: " << SAGEATTN4_DEBUG_THREAD_IDX << "\n\n";

  std::cout << "Types\n";
  print_type<Element>("Element");
  print_type<ElementSF>("ElementSF");
  print_type<typename Traits::ElementOut>("ElementOut");
  print_type<typename Traits::ElementQMma>("ElementQMma");
  print_type<typename Traits::ElementKMma>("ElementKMma");
  print_type<typename Traits::TiledMmaQK::Atom>("TiledMmaQK::Atom");
  print_type<typename Traits::TiledMmaPV::Atom>("TiledMmaPV::Atom");
  print_type<typename Mainloop::TMA_Q>("Mainloop::TMA_Q");
  print_type<typename Mainloop::TMA_KV>("Mainloop::TMA_KV");
  print_type<typename Mainloop::TMA_Vt>("Mainloop::TMA_Vt");
  std::cout << "\n";

  std::cout << "Core shapes\n";
  print_cute("TileShape_MNK", TileShape_MNK{});
  print_cute("ClusterShape_MNK", typename Traits::ClusterShape_MNK{});
  print_cute("AtomLayoutMNK", typename Traits::AtomLayoutMNK{});
  print_cute("PermTileM", typename Traits::PermTileM{});
  print_cute("PermTileN", typename Traits::PermTileN{});
  print_cute("PermTileK", typename Traits::PermTileK{});
  std::cout << "  NumSFQK: " << Traits::NumSFQK << "\n";
  std::cout << "  NumSFPV: " << Traits::NumSFPV << "\n";
  std::cout << "  MMA_NSF: " << Traits::MMA_NSF << "\n\n";

  typename Traits::TiledMmaQK tiled_mma_qk;
  typename Traits::TiledMmaPV tiled_mma_pv;
  Mainloop mainloop;

  std::cout << "MMA QK\n";
  print_cute("TiledMmaQK", tiled_mma_qk);
  print_cute("tile_shape(QK)", cute::tile_shape(tiled_mma_qk));
  print_cute("AtomShape_MNK(QK)", typename Traits::TiledMmaQK::AtomShape_MNK{});
  print_cute("AtomLayoutA_TV(QK)", typename Traits::TiledMmaQK::AtomLayoutA_TV{});
  print_cute("AtomLayoutB_TV(QK)", typename Traits::TiledMmaQK::AtomLayoutB_TV{});
  print_cute("AtomLayoutC_TV(QK)", typename Traits::TiledMmaQK::AtomLayoutC_TV{});
  print_cute("thr_layout_vmnk(QK)", tiled_mma_qk.get_thr_layout_vmnk());
  print_cute("layoutSFA_TV(QK)", mainloop.get_layoutSFA_TV(tiled_mma_qk));
  print_cute("layoutSFB_TV(QK)", mainloop.get_layoutSFB_TV(tiled_mma_qk));
  std::cout << "\n";

  std::cout << "MMA PV\n";
  print_cute("TiledMmaPV", tiled_mma_pv);
  print_cute("tile_shape(PV)", cute::tile_shape(tiled_mma_pv));
  print_cute("AtomShape_MNK(PV)", typename Traits::TiledMmaPV::AtomShape_MNK{});
  print_cute("thr_layout_vmnk(PV)", tiled_mma_pv.get_thr_layout_vmnk());
  print_cute("layoutSFB_TV(PV)", mainloop.get_layoutSFB_TV(tiled_mma_pv));
  std::cout << "\n";

  std::cout << "Shared memory storage\n";
  std::cout << "  SharedStorage bytes: " << sizeof(typename Traits::SharedStorage) << "\n";
  std::cout << "  SharedStorage align: " << alignof(typename Traits::SharedStorage) << "\n";
  std::cout << "  smem_q elements: " << cute::cosize_v<typename Traits::SmemLayoutQ> << "\n";
  std::cout << "  smem_k elements: " << cute::cosize_v<typename Traits::SmemLayoutK> << "\n";
  std::cout << "  smem_v elements: " << cute::cosize_v<typename Traits::SmemLayoutV> << "\n";
  std::cout << "  smem_ds elements: " << cute::cosize_v<typename Traits::SmemLayoutDS> << "\n";
  std::cout << "  smem_o elements: " << cute::cosize_v<typename Traits::SmemLayoutO> << "\n";
  std::cout << "  smem_SFQ elements: " << cute::cosize_v<typename Traits::SmemLayoutSFQ> << "\n";
  std::cout << "  smem_SFK elements: " << cute::cosize_v<typename Traits::SmemLayoutSFK> << "\n";
  std::cout << "  smem_SFVt elements: " << cute::cosize_v<typename Traits::SmemLayoutSFVt> << "\n\n";

  dump_layout<typename Traits::SmemLayoutQ>("SmemLayoutQ");
  dump_layout<typename Traits::SmemLayoutK>("SmemLayoutK");
  dump_layout<typename Traits::SmemLayoutV>("SmemLayoutV");
  dump_layout<typename Traits::SmemLayoutVt>("SmemLayoutVt");
  dump_layout<typename Traits::SmemLayoutDS>("SmemLayoutDS");
  dump_layout<typename Traits::SmemLayoutO>("SmemLayoutO");
  dump_layout<typename Traits::SmemLayoutSFQ>("SmemLayoutSFQ");
  dump_layout<typename Traits::SmemLayoutSFK>("SmemLayoutSFK");
  dump_layout<typename Traits::SmemLayoutSFV>("SmemLayoutSFV");
  dump_layout<typename Traits::SmemLayoutSFVt>("SmemLayoutSFVt");
  dump_layout<typename Traits::LayoutP>("LayoutP");
  dump_layout<typename Traits::LayoutSFP>("LayoutSFP");
  std::cout << "\n";

  std::cout << "TMA transaction bytes\n";
  std::cout << "  Q: " << Mainloop::TmaTransactionBytesQ << "\n";
  std::cout << "  K: " << Mainloop::TmaTransactionBytesK << "\n";
  std::cout << "  V: " << Mainloop::TmaTransactionBytesV << "\n\n";

  auto sQ = cute::make_tensor(cute::make_smem_ptr<Element>(nullptr), typename Traits::SmemLayoutQ{});
  auto sK = cute::make_tensor(cute::make_smem_ptr<Element>(nullptr), typename Traits::SmemLayoutK{});
  auto sVt = cute::make_tensor(cute::make_smem_ptr<Element>(nullptr), typename Traits::SmemLayoutVt{});
  auto sDS = cute::make_tensor(cute::make_smem_ptr<float>(nullptr), typename Traits::SmemLayoutDS{});
  auto sSFQ = cute::make_tensor(cute::make_smem_ptr<ElementSF>(nullptr), typename Traits::SmemLayoutSFQ{});
  auto sSFK = cute::make_tensor(cute::make_smem_ptr<ElementSF>(nullptr), typename Traits::SmemLayoutSFK{});
  auto sSFVt = cute::make_tensor(cute::make_smem_ptr<ElementSF>(nullptr), typename Traits::SmemLayoutSFVt{});

  auto thread_mma_qk = tiled_mma_qk.get_thread_slice(SAGEATTN4_DEBUG_THREAD_IDX);
  auto thread_mma_pv = tiled_mma_pv.get_thread_slice(SAGEATTN4_DEBUG_THREAD_IDX);

  auto tSrQ = thread_mma_qk.partition_fragment_A(sQ);
  auto tSrK = thread_mma_qk.partition_fragment_B(sK(cute::_, cute::_, cute::Int<0>{}));
  auto tOrVt = thread_mma_pv.partition_fragment_B(sVt(cute::_, cute::_, cute::Int<0>{}));
  auto tSrS = cute::partition_fragment_C(tiled_mma_qk, cute::select<0, 1>(TileShape_MNK{}));
  auto tOrO = cute::partition_fragment_C(tiled_mma_pv, cute::select<0, 2>(TileShape_MNK{}));
  auto tSrSFQ = mainloop.partition_fragment_SFA(sSFQ, thread_mma_qk);
  auto tSrSFK = mainloop.partition_fragment_SFB(sSFK(cute::_, cute::_, cute::Int<0>{}), thread_mma_qk);
  auto tOrSFVt = mainloop.partition_fragment_SFB(sSFVt(cute::_, cute::_, cute::Int<0>{}), thread_mma_pv);
  auto tOrP = cute::make_tensor<Element>(typename Traits::LayoutP{});
  auto tOrSFP = cute::make_tensor<ElementSF>(typename Traits::LayoutSFP{});
  auto tSrDS = cute::make_tensor<float>(cute::make_shape(cute::_8{}, cute::_4{}),
                                        cute::make_stride(cute::_1{}, cute::_8{}));
  auto tSrSConversion = cute::make_tensor(tSrS.data(), debug_convert_to_conversion_layout(tSrS.layout()));
  auto AbsMaxP = cute::make_tensor_like<float>(
      cute::make_layout(cute::shape(cute::group<1, 4>(
          cute::flatten(tSrSConversion.layout()(cute::make_coord(cute::_0{}, cute::_), cute::_, cute::_))))));

  std::cout << "Mainloop consumer fragments\n";
  dump_tensor("sQ", sQ);
  dump_tensor("sK", sK);
  dump_tensor("sVt", sVt);
  dump_tensor("sDS", sDS);
  dump_tensor("sSFQ", sSFQ);
  dump_tensor("sSFK", sSFK);
  dump_tensor("sSFVt", sSFVt);
  dump_tensor("tSrQ", tSrQ);
  dump_tensor("tSrK", tSrK);
  dump_tensor("tOrVt", tOrVt);
  dump_tensor("tSrS", tSrS);
  dump_tensor("tSrSConversion", tSrSConversion);
  dump_tensor("AbsMaxP", AbsMaxP);
  dump_tensor("tOrO", tOrO);
  dump_tensor("tSrSFQ", tSrSFQ);
  dump_tensor("tSrSFK", tSrSFK);
  dump_tensor("tOrSFVt", tOrSFVt);
  dump_tensor("tOrP", tOrP);
  dump_tensor("tOrSFP", tOrSFP);
  dump_tensor("tSrDS", tSrDS);
  std::cout << "\n";

  auto smem_tiled_copy_Q = cute::make_tiled_copy_A(typename Traits::SmemCopyAtomQ{}, tiled_mma_qk);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(SAGEATTN4_DEBUG_THREAD_IDX);
  auto tSsQ = smem_thr_copy_Q.partition_S(cute::as_position_independent_swizzle_tensor(sQ));
  auto tSrQCopyView = smem_thr_copy_Q.retile_D(tSrQ);

  auto smem_tiled_copy_K = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomKV{}, tiled_mma_qk);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(SAGEATTN4_DEBUG_THREAD_IDX);
  auto tSsK = smem_thr_copy_K.partition_S(cute::as_position_independent_swizzle_tensor(sK));
  auto tSrKCopyView = smem_thr_copy_K.retile_D(tSrK);

  auto smem_tiled_copy_V = cute::make_tiled_copy_B(typename Traits::SmemCopyAtomKV{}, tiled_mma_pv);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(SAGEATTN4_DEBUG_THREAD_IDX);
  auto tOsVt = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sVt));
  auto tOrVtCopyView = smem_thr_copy_V.retile_D(tOrVt);

  std::cout << "Mainloop smem copy views\n";
  print_cute("smem_tiled_copy_Q", smem_tiled_copy_Q);
  dump_tensor("tSsQ", tSsQ);
  dump_tensor("tSrQCopyView", tSrQCopyView);
  print_cute("smem_tiled_copy_K", smem_tiled_copy_K);
  dump_tensor("tSsK", tSsK);
  dump_tensor("tSrKCopyView", tSrKCopyView);
  print_cute("smem_tiled_copy_V", smem_tiled_copy_V);
  dump_tensor("tOsVt", tOsVt);
  dump_tensor("tOrVtCopyView", tOrVtCopyView);
}

}  // namespace

int main(int argc, char** argv) {
  using OutputType = cutlass::bfloat16_t;
  using Traits = Flash_fwd_kernel_traits<
      SAGEATTN4_DEBUG_HEADDIM,
      SAGEATTN4_DEBUG_BLOCK_M,
      SAGEATTN4_DEBUG_BLOCK_N,
      SAGEATTN4_DEBUG_STAGES,
      SAGEATTN4_DEBUG_CLUSTER_M,
      (SAGEATTN4_DEBUG_BLOCK_MEAN != 0),
      cutlass::nv_float4_t<cutlass::float_e2m1_t>,
      OutputType>;

  bool emit_access_map_csv = argc > 1 && std::string(argv[1]) == "--access-map-csv";
  dump_mainloop_traits<Traits, (SAGEATTN4_DEBUG_IS_CAUSAL != 0)>(emit_access_map_csv);
  return 0;
}
