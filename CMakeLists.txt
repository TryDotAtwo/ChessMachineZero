cmake_minimum_required(VERSION 3.22)
project(cmz_pure_vm LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
include(CTest)

add_library(cmz_pure_vm STATIC src/artifact.cpp src/hullkv.cu src/hullkv_autograd.cpp src/module.cpp src/ste.cpp)
target_include_directories(cmz_pure_vm PUBLIC include)
target_compile_features(cmz_pure_vm PUBLIC cxx_std_17)
set_target_properties(cmz_pure_vm PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(cmz_pure_vm PUBLIC "${TORCH_LIBRARIES}")

add_executable(cmz_recurrent_test tests/native/recurrent_test.cpp)
target_link_libraries(cmz_recurrent_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_recurrent_test COMMAND cmz_recurrent_test)

add_executable(cmz_artifact_test tests/native/artifact_test.cpp)
target_link_libraries(cmz_artifact_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_artifact_test COMMAND cmz_artifact_test)

add_executable(cmz_ste_test tests/native/ste_test.cpp)
target_link_libraries(cmz_ste_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_ste_test COMMAND cmz_ste_test)

add_executable(cmz_hullkv_test tests/native/hullkv_test.cpp)
target_link_libraries(cmz_hullkv_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_hullkv_test COMMAND cmz_hullkv_test)

add_executable(cmz_position_artifact_test tests/native/position_artifact_test.cpp)
target_link_libraries(cmz_position_artifact_test PRIVATE cmz_pure_vm)
