cmake_minimum_required(VERSION 3.22)
project(cmz_pure_vm LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
include(CTest)

add_library(cmz_pure_vm STATIC src/artifact.cpp src/hullkv.cu src/hullkv_autograd.cpp src/module.cpp src/ste.cpp)
target_include_directories(cmz_pure_vm PUBLIC include)
target_compile_features(cmz_pure_vm PUBLIC cxx_std_17)
set_target_properties(cmz_pure_vm PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(cmz_pure_vm PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(cmz_pure_vm PUBLIC CUDA::cudart)
if(MSVC)
    target_compile_options(cmz_pure_vm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    # Keep CUDA registration when linking against Torch's Windows import libraries.
    target_link_options(cmz_pure_vm INTERFACE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

add_executable(cmz_recurrent_test tests/native/recurrent_test.cpp)
target_link_libraries(cmz_recurrent_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_recurrent_test COMMAND cmz_recurrent_test)

add_executable(cmz_artifact_test tests/native/artifact_test.cpp)
target_link_libraries(cmz_artifact_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_artifact_test COMMAND cmz_artifact_test)

add_executable(cmz_ste_test tests/native/ste_test.cpp)
target_link_libraries(cmz_ste_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_ste_test COMMAND cmz_ste_test)
foreach(case empty-mask-cpu empty-mask-cuda invalid-temperature invalid-scale)
    add_test(NAME cmz_ste_${case} COMMAND cmz_ste_test ${case})
endforeach()

add_executable(cmz_hullkv_test tests/native/hullkv_test.cpp)
target_link_libraries(cmz_hullkv_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_hullkv_test COMMAND cmz_hullkv_test)
foreach(case metadata index-negative index-large index-negative-batched index-large-batched)
    # Each bad GPU input has its own process; an asserted context cannot be reused.
    add_test(NAME cmz_hullkv_${case} COMMAND cmz_hullkv_test ${case})
endforeach()

add_executable(cmz_graph_validation_test tests/native/graph_validation_test.cpp)
target_link_libraries(cmz_graph_validation_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_graph_validation_test COMMAND cmz_graph_validation_test)

add_executable(cmz_position_artifact_test tests/native/position_artifact_test.cpp)
target_link_libraries(cmz_position_artifact_test PRIVATE cmz_pure_vm)
set(CMZ_POSITION_ARTIFACT "" CACHE FILEPATH "Generated position.cmz artifact")
set(CMZ_POSITION_FIXTURES "" CACHE FILEPATH "Generated positions.bin oracle fixtures")
if(EXISTS "${CMZ_POSITION_ARTIFACT}" AND EXISTS "${CMZ_POSITION_FIXTURES}")
    add_test(NAME cmz_position_artifact_test COMMAND cmz_position_artifact_test
        "${CMZ_POSITION_ARTIFACT}" "${CMZ_POSITION_FIXTURES}")
else()
    message(STATUS "Position acceptance NOT registered: set CMZ_POSITION_ARTIFACT and CMZ_POSITION_FIXTURES; see README.md")
endif()
