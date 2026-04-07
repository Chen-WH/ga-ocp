cmake_minimum_required(VERSION 3.14)
project(ga_ocp_core)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()
set(COAL_DISABLE_HPP_FCL_WARNINGS TRUE)
add_compile_definitions(COAL_DISABLE_HPP_FCL_WARNINGS)

# Debug configuration
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")

# Release configuration
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GA_OCP_BUILD_CROCODDYL_TESTS "Build Crocoddyl integration tests" ON)
option(GA_OCP_BUILD_BENCHMARKS "Build Crocoddyl benchmarks" OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TetraPGA CONFIG REQUIRED)
find_package(pinocchio REQUIRED)
find_package(crocoddyl REQUIRED)

if(NOT TetraPGA_DESCRIPTION_DIR)
    message(FATAL_ERROR "TetraPGA_DESCRIPTION_DIR was not exported by TetraPGAConfig.cmake")
endif()

if(GA_OCP_BUILD_CROCODDYL_TESTS OR GA_OCP_BUILD_BENCHMARKS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(COAL REQUIRED IMPORTED_TARGET coal)
endif()

if(GA_OCP_BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    find_package(mim_solvers CONFIG REQUIRED)
    pkg_check_modules(CASADI REQUIRED casadi)
endif()

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
)

set(GA_OCP_CROCODDYL_TEST_TARGETS)
set(GA_OCP_BENCHMARK_TARGETS)

function(ga_ocp_add_crocoddyl_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name}
        TetraPGA::TetraPGA
        crocoddyl::crocoddyl
        pinocchio::pinocchio
    )
    target_compile_definitions(${target_name}
        PRIVATE GA_OCP_TETRAPGA_DESCRIPTION_DIR="${TetraPGA_DESCRIPTION_DIR}")
    list(APPEND GA_OCP_CROCODDYL_TEST_TARGETS ${target_name})
    set(GA_OCP_CROCODDYL_TEST_TARGETS "${GA_OCP_CROCODDYL_TEST_TARGETS}" PARENT_SCOPE)
endfunction()

function(ga_ocp_add_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name}
        TetraPGA::TetraPGA
        benchmark::benchmark
        crocoddyl::crocoddyl
        pinocchio::pinocchio
        ${CASADI_LIBRARIES}
    )
    target_compile_definitions(${target_name}
        PRIVATE
          GA_OCP_TETRAPGA_DESCRIPTION_DIR="${TetraPGA_DESCRIPTION_DIR}"
          GA_OCP_HAS_CASADI_BENCH=1)
    target_include_directories(${target_name} PRIVATE ${CASADI_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE ${CASADI_LIBRARY_DIRS})
    list(APPEND GA_OCP_BENCHMARK_TARGETS ${target_name})
    set(GA_OCP_BENCHMARK_TARGETS "${GA_OCP_BENCHMARK_TARGETS}" PARENT_SCOPE)
endfunction()

if(GA_OCP_BUILD_CROCODDYL_TESTS)
    ga_ocp_add_crocoddyl_test(Crocoddyl_fddp_test test/Crocoddyl_fddp_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_torLim_test test/Crocoddyl_torLim_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_velLim_test test/Crocoddyl_velLim_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_motor_test test/Crocoddyl_motor_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_collision_test test/Crocoddyl_collision_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_idmpc_test test/Crocoddyl_idmpc_test.cpp)
    target_link_libraries(Crocoddyl_collision_test PkgConfig::COAL)
endif()

if(GA_OCP_BUILD_BENCHMARKS)
    ga_ocp_add_benchmark(Crocoddyl_fddp_bench benchmark/Crocoddyl_fddp_bench.cpp)
    ga_ocp_add_benchmark(Crocoddyl_fddp_budget_bench benchmark/Crocoddyl_fddp_budget_bench.cpp)
    ga_ocp_add_benchmark(Crocoddyl_sqp_bench benchmark/Crocoddyl_sqp_bench.cpp)
    target_link_libraries(Crocoddyl_sqp_bench mim_solvers::mim_solvers)
endif()

if(GA_OCP_BUILD_CROCODDYL_TESTS)
    install(TARGETS ${GA_OCP_CROCODDYL_TEST_TARGETS} DESTINATION lib/${PROJECT_NAME})
endif()

if(GA_OCP_BUILD_BENCHMARKS)
    install(TARGETS ${GA_OCP_BENCHMARK_TARGETS} DESTINATION lib/${PROJECT_NAME})
endif()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY "${TetraPGA_DESCRIPTION_DIR}/" DESTINATION share/${PROJECT_NAME}/description)

ament_export_include_directories(include)
ament_export_dependencies(Eigen3 TetraPGA pinocchio crocoddyl)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
