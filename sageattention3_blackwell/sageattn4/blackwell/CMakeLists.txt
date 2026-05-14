cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(SAGEATTN3_NVCC nvcc HINTS "$ENV{CUDA_HOME}/bin" /usr/local/cuda/bin)
  if(SAGEATTN3_NVCC)
    set(CMAKE_CUDA_COMPILER "${SAGEATTN3_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()

project(sageattn4_blackwell LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)

get_filename_component(SAGEATTN3_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(REPO_ROOT "${SAGEATTN3_DIR}/.." ABSOLUTE)

set(CUTLASS_DIR "${REPO_ROOT}/csrc/cutlass" CACHE PATH "CUTLASS checkout used by setup.py")
set(SAGEATTN3_CUDA_ARCH "compute_120a,code=sm_120a" CACHE STRING "NVCC -gencode arch/code pair")
set(SAGEATTN3_PYTHON_EXECUTABLE "/venv/main/bin/python" CACHE FILEPATH "Python executable used to query PyTorch")
set(SAGEATTN4_CUDA_FAST_COMPILE "mid" CACHE STRING "nvcc --Ofast-compile level for CMake builds: 0, min, mid, or max")

if(NOT EXISTS "${SAGEATTN3_PYTHON_EXECUTABLE}")
  message(FATAL_ERROR "Expected Python executable at ${SAGEATTN3_PYTHON_EXECUTABLE}")
endif()
set(Python3_EXECUTABLE "${SAGEATTN3_PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable" FORCE)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)
find_path(SAGEATTN4_CUDA_CCCL_INCLUDE_DIR
  cuda/std/utility
  HINTS
    "${CUDAToolkit_TARGET_DIR}/include/cccl"
    "${CUDAToolkit_INCLUDE_DIRS}/cccl"
    "/usr/local/cuda/targets/x86_64-linux/include/cccl"
    "/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl"
    "/usr/local/cuda-12.9/targets/x86_64-linux/include/cccl"
)

set(TORCH_INCLUDE_DIRS "")
set(TORCH_LIBRARY_DIRS "")
set(NVIDIA_CUDA_PACKAGE_INCLUDE_DIR "")
set(PYTHON_EXTENSION_SUFFIX "")

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths(device_type='cuda')))"
  OUTPUT_VARIABLE TORCH_LIBRARY_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "from pathlib import Path; import site; paths = site.getsitepackages() + [site.getusersitepackages()]; hits = [str(Path(p) / 'nvidia' / 'cu13' / 'include') for p in paths if (Path(p) / 'nvidia' / 'cu13' / 'include').exists()]; print(hits[0] if hits else '')"
  OUTPUT_VARIABLE NVIDIA_CUDA_PACKAGE_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(SAGEATTN4_GENERATED_DIR "${CMAKE_CURRENT_LIST_DIR}/generated")
set(SAGEATTN4_GENERATED_HEADER "${SAGEATTN4_GENERATED_DIR}/sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean1.h")
set(SAGEATTN4_GENERATED_STAMP "${CMAKE_CURRENT_BINARY_DIR}/sageattn4_generated_headers.stamp")
set(SAGEATTN4_SPECIALIZED_KERNEL_SOURCE "${CMAKE_CURRENT_LIST_DIR}/fwd_kernel.cu")
add_custom_command(
  OUTPUT "${SAGEATTN4_GENERATED_STAMP}"
  BYPRODUCTS "${SAGEATTN4_GENERATED_HEADER}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/generate_specialized_headers.py"
          --output-dir "${SAGEATTN4_GENERATED_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${SAGEATTN4_GENERATED_STAMP}"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/generate_specialized_headers.py"
          "${CMAKE_CURRENT_LIST_DIR}/specialized_codegen.py"
          "${CMAKE_CURRENT_LIST_DIR}/generated_templates/sageattn4_fwd_specialization.h.in"
  COMMENT "Generating SageAttention4 Blackwell specialized type headers"
  VERBATIM
)
add_custom_target(sageattn4_generated_headers DEPENDS "${SAGEATTN4_GENERATED_STAMP}")

set(NVIDIA_CUDA_LIBRARY_HEADER_SHIM_DIR "")
if(NVIDIA_CUDA_PACKAGE_INCLUDE_DIR)
  set(NVIDIA_CUDA_LIBRARY_HEADER_SHIM_DIR "${CMAKE_CURRENT_BINARY_DIR}/nvidia_cuda_library_headers")
  file(MAKE_DIRECTORY "${NVIDIA_CUDA_LIBRARY_HEADER_SHIM_DIR}")
  file(GLOB NVIDIA_CUDA_LIBRARY_HEADERS CONFIGURE_DEPENDS
    "${NVIDIA_CUDA_PACKAGE_INCLUDE_DIR}/cublas*.h"
    "${NVIDIA_CUDA_PACKAGE_INCLUDE_DIR}/cusolver*.h"
    "${NVIDIA_CUDA_PACKAGE_INCLUDE_DIR}/cusparse*.h"
  )
  foreach(NVIDIA_CUDA_LIBRARY_HEADER IN LISTS NVIDIA_CUDA_LIBRARY_HEADERS)
    get_filename_component(NVIDIA_CUDA_LIBRARY_HEADER_NAME "${NVIDIA_CUDA_LIBRARY_HEADER}" NAME)
    file(CREATE_LINK
      "${NVIDIA_CUDA_LIBRARY_HEADER}"
      "${NVIDIA_CUDA_LIBRARY_HEADER_SHIM_DIR}/${NVIDIA_CUDA_LIBRARY_HEADER_NAME}"
      SYMBOLIC COPY_ON_ERROR
    )
  endforeach()
endif()

file(GLOB BLACKWELL_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/generated/*.h"
)
file(GLOB QUANTIZATION_HEADERS CONFIGURE_DEPENDS "${SAGEATTN3_DIR}/quantization/*.h")

function(configure_sageattn4_cuda_extension TARGET_NAME)
  set_target_properties(${TARGET_NAME} PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
    CUDA_ARCHITECTURES OFF
  )

  target_include_directories(${TARGET_NAME} PRIVATE
    "${SAGEATTN3_DIR}"
    "${SAGEATTN3_DIR}/blackwell"
    "${SAGEATTN3_DIR}/quantization"
    "${CUTLASS_DIR}/include"
    "${CUTLASS_DIR}/tools/util/include"
    ${NVIDIA_CUDA_LIBRARY_HEADER_SHIM_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${SAGEATTN4_CUDA_CCCL_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
  )

  target_link_directories(${TARGET_NAME} PRIVATE
    ${TORCH_LIBRARY_DIRS}
  )

  target_link_libraries(${TARGET_NAME} PRIVATE
    c10
    torch
    torch_cpu
    torch_python
    c10_cuda
    torch_cuda
    CUDA::cudart
    CUDA::cuda_driver
  )

  foreach(CUDAToolkit_INCLUDE_DIR IN LISTS CUDAToolkit_INCLUDE_DIRS)
    target_compile_options(${TARGET_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-I${CUDAToolkit_INCLUDE_DIR}>
    )
  endforeach()

  target_compile_definitions(${TARGET_NAME} PRIVATE
    TORCH_API_INCLUDE_EXTENSION_H
    TORCH_EXTENSION_NAME=${TARGET_NAME}
    CUTLASS_DEBUG_TRACE_LEVEL=0
    NDEBUG
    QBLKSIZE=128
    KBLKSIZE=128
    CTA256
    DQINRMEM
    EXECMODE=0
  )
  target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=${SAGEATTN3_CUDA_ARCH}>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=4>
  )
  if(NOT SAGEATTN4_CUDA_FAST_COMPILE STREQUAL "0")
    target_compile_options(${TARGET_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=${SAGEATTN4_CUDA_FAST_COMPILE}>
    )
  endif()
endfunction()

add_library(fp4attn4_cuda MODULE
  "${CMAKE_CURRENT_LIST_DIR}/api.cpp"
  "${SAGEATTN4_SPECIALIZED_KERNEL_SOURCE}"
  ${BLACKWELL_HEADERS}
)
add_dependencies(fp4attn4_cuda sageattn4_generated_headers)
configure_sageattn4_cuda_extension(fp4attn4_cuda)

add_library(fp4quant4_cuda MODULE
  "${SAGEATTN3_DIR}/quantization/fp4_quantization_4d.cu"
  ${QUANTIZATION_HEADERS}
)
configure_sageattn4_cuda_extension(fp4quant4_cuda)
