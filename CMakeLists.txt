cmake_minimum_required(VERSION 3.22)
project(cmz_pure_vm LANGUAGES CXX)

find_package(Torch REQUIRED)
include(CTest)

add_library(cmz_pure_vm STATIC src/module.cpp)
target_include_directories(cmz_pure_vm PUBLIC include)
target_compile_features(cmz_pure_vm PUBLIC cxx_std_17)
target_link_libraries(cmz_pure_vm PUBLIC "${TORCH_LIBRARIES}")

add_executable(cmz_recurrent_test tests/native/recurrent_test.cpp)
target_link_libraries(cmz_recurrent_test PRIVATE cmz_pure_vm)
add_test(NAME cmz_recurrent_test COMMAND cmz_recurrent_test)
