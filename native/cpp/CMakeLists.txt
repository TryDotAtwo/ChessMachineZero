cmake_minimum_required(VERSION 3.22)
project(cmz_engine LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

set(CUTLASS_INCLUDE_DIR "/opt/cutlass/include" CACHE PATH "CUTLASS include directory")
if(NOT EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass/gemm/device/gemm.h")
    message(FATAL_ERROR "CUTLASS headers are required at ${CUTLASS_INCLUDE_DIR}; no fallback is allowed")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(cmz_engine STATIC
    src/cmz_engine.cpp
    src/cmz_cuda_kernels.cu
)

target_include_directories(cmz_engine PUBLIC include PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIR})
target_link_libraries(cmz_engine PUBLIC CUDA::cudart ${TORCH_LIBRARIES})
target_compile_options(cmz_engine PRIVATE "${TORCH_CXX_FLAGS}")
set_target_properties(cmz_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
