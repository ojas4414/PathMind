cmake_minimum_required(VERSION 3.15)
project(traffic_env)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)

pybind11_add_module(traffic_env
    cpp/TrafficEnvironment.cpp
    cpp/bindings.cpp
)

target_include_directories(traffic_env PRIVATE cpp/)