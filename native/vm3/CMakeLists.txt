cmake_minimum_required(VERSION 3.22)
project(cmz_vm3 LANGUAGES CXX)

find_package(Torch REQUIRED)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# These explicit manifests are consumed by the purity auditor. Every future
# runtime source must belong to exactly one target closure.
set(CMZ_VM3_EXECUTOR_SOURCES
)
set(CMZ_VM3_POLICY_SOURCES
  src/st_selector.cpp
)
set(CMZ_VM3_LOADER_SOURCES
  src/program_loader.cpp
)
set(CMZ_VM3_COMPILER_SOURCES
)

add_library(cmz_vm3_program_loader STATIC ${CMZ_VM3_LOADER_SOURCES})
target_include_directories(cmz_vm3_program_loader PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(cmz_vm3_program_loader PUBLIC ${TORCH_LIBRARIES})

add_library(cmz_vm3_policy STATIC ${CMZ_VM3_POLICY_SOURCES})
target_include_directories(cmz_vm3_policy PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(cmz_vm3_policy PUBLIC ${TORCH_LIBRARIES})

add_library(cmz_vm3_compiler INTERFACE)
target_include_directories(cmz_vm3_compiler INTERFACE include)

add_library(cmz_vm3_executor INTERFACE)
target_include_directories(cmz_vm3_executor INTERFACE include)
target_link_libraries(cmz_vm3_executor INTERFACE cmz_vm3_policy cmz_vm3_program_loader)

add_executable(cmz_vm3_contract tests/test_contract.cpp)
target_include_directories(cmz_vm3_contract PRIVATE include)
add_test(NAME cmz_vm3_contract COMMAND cmz_vm3_contract)

add_executable(cmz_vm3_test_program_loader tests/test_program_loader.cpp)
target_link_libraries(cmz_vm3_test_program_loader PRIVATE cmz_vm3_program_loader ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_program_loader COMMAND cmz_vm3_test_program_loader)

add_executable(cmz_vm3_test_st_selector tests/test_st_selector.cpp)
target_link_libraries(cmz_vm3_test_st_selector PRIVATE cmz_vm3_executor ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_st_selector COMMAND cmz_vm3_test_st_selector)
