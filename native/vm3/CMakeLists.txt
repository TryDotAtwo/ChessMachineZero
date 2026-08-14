cmake_minimum_required(VERSION 3.22)
project(cmz_vm3 LANGUAGES CXX)

find_package(Torch REQUIRED)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# These explicit manifests are consumed by the purity auditor. Every future
# runtime source must belong to exactly one target closure.
set(CMZ_VM3_EXECUTOR_SOURCES
  src/attention2d.cpp
  src/chess_state_views.cpp
  src/recurrent_ring.cpp
  src/trajectory.cpp
)
set(CMZ_VM3_POLICY_SOURCES
  src/policy.cpp
  src/st_selector.cpp
)
set(CMZ_VM3_LOADER_SOURCES
  src/policy_preflight.cpp
  src/program_loader.cpp
)
set(CMZ_VM3_COMPILER_SOURCES
  compiler/candidate_bank.cpp
  compiler/initial_state_binder.cpp
  compiler/pawn_rules.cpp
  compiler/piece_rules.cpp
  compiler/rule_image_writer.cpp
  compiler/rule_compiler.cpp
)

add_library(cmz_vm3_program_loader STATIC ${CMZ_VM3_LOADER_SOURCES})
target_include_directories(cmz_vm3_program_loader PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(cmz_vm3_program_loader PUBLIC ${TORCH_LIBRARIES})

add_library(cmz_vm3_policy STATIC ${CMZ_VM3_POLICY_SOURCES})
target_include_directories(cmz_vm3_policy PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(cmz_vm3_policy PUBLIC ${TORCH_LIBRARIES})

add_library(cmz_vm3_compiler STATIC ${CMZ_VM3_COMPILER_SOURCES})
target_include_directories(cmz_vm3_compiler PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(cmz_vm3_compiler PUBLIC cmz_vm3_program_loader ${TORCH_LIBRARIES})

add_library(cmz_vm3_executor STATIC ${CMZ_VM3_EXECUTOR_SOURCES})
target_include_directories(cmz_vm3_executor PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(cmz_vm3_executor PUBLIC cmz_vm3_policy cmz_vm3_program_loader ${TORCH_LIBRARIES})

add_executable(cmz_vm3_compile_minimal_rules compiler/rule_compiler_main.cpp)
target_link_libraries(cmz_vm3_compile_minimal_rules PRIVATE cmz_vm3_compiler ${TORCH_LIBRARIES})

add_executable(cmz_vm3_contract tests/test_contract.cpp)
target_include_directories(cmz_vm3_contract PRIVATE include)
add_test(NAME cmz_vm3_contract COMMAND cmz_vm3_contract)

add_executable(cmz_vm3_test_program_loader tests/test_program_loader.cpp)
target_link_libraries(cmz_vm3_test_program_loader PRIVATE cmz_vm3_program_loader ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_program_loader COMMAND cmz_vm3_test_program_loader)

add_executable(cmz_vm3_test_st_selector tests/test_st_selector.cpp)
target_link_libraries(cmz_vm3_test_st_selector PRIVATE cmz_vm3_executor ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_st_selector COMMAND cmz_vm3_test_st_selector)

add_executable(cmz_vm3_test_attention2d tests/test_attention2d.cpp)
target_link_libraries(cmz_vm3_test_attention2d PRIVATE cmz_vm3_executor ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_attention2d COMMAND cmz_vm3_test_attention2d)

add_executable(cmz_vm3_test_state_schema tests/test_state_schema.cpp)
target_link_libraries(cmz_vm3_test_state_schema PRIVATE cmz_vm3_executor cmz_vm3_compiler ${TORCH_LIBRARIES})
add_executable(cmz_vm3_state_oracle_cli tests/state_oracle_cli.cpp)
target_link_libraries(cmz_vm3_state_oracle_cli PRIVATE cmz_vm3_executor cmz_vm3_compiler ${TORCH_LIBRARIES})
add_dependencies(cmz_vm3_test_state_schema cmz_vm3_state_oracle_cli)
add_test(NAME cmz_vm3_state_schema COMMAND cmz_vm3_test_state_schema)

add_executable(cmz_vm3_test_trajectory tests/test_trajectory.cpp)
target_link_libraries(cmz_vm3_test_trajectory PRIVATE cmz_vm3_executor ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_trajectory COMMAND cmz_vm3_test_trajectory)

add_executable(cmz_vm3_test_candidate_bank tests/test_candidate_bank.cpp)
target_link_libraries(cmz_vm3_test_candidate_bank PRIVATE cmz_vm3_compiler ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_candidate_bank COMMAND cmz_vm3_test_candidate_bank)

add_executable(cmz_vm3_test_policy tests/test_policy.cpp)
target_link_libraries(cmz_vm3_test_policy PRIVATE cmz_vm3_executor ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_policy COMMAND cmz_vm3_test_policy)

add_executable(cmz_vm3_test_minimal_ring tests/test_minimal_ring.cpp)
target_link_libraries(cmz_vm3_test_minimal_ring PRIVATE cmz_vm3_executor cmz_vm3_compiler ${TORCH_LIBRARIES})
add_test(NAME cmz_vm3_minimal_ring COMMAND cmz_vm3_test_minimal_ring)

add_executable(cmz_vm3_test_pseudo_legal tests/test_pseudo_legal.cpp)
target_link_libraries(cmz_vm3_test_pseudo_legal PRIVATE cmz_vm3_executor cmz_vm3_compiler ${TORCH_LIBRARIES})
add_executable(cmz_vm3_pseudo_legal_oracle_cli tests/pseudo_legal_oracle_cli.cpp)
target_link_libraries(cmz_vm3_pseudo_legal_oracle_cli PRIVATE cmz_vm3_executor cmz_vm3_compiler ${TORCH_LIBRARIES})
add_dependencies(cmz_vm3_test_pseudo_legal cmz_vm3_pseudo_legal_oracle_cli)
add_test(NAME cmz_vm3_pseudo_legal COMMAND cmz_vm3_test_pseudo_legal)
