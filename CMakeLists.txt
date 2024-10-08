cmake_minimum_required(VERSION 3.30)

project(pycpp)

set(Python_FIND_VIRTUALENV FIRST)
find_package(pybind11 REQUIRED)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR}/.local)

find_package(OpenBLAS REQUIRED)
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/eigen-3.4.0)


pybind11_add_module(pycppmod mod.cc)

target_link_libraries(pycppmod PRIVATE pybind11::pybind11 LAPACK::LAPACK BLAS::BLAS)
target_include_directories(pycppmod PRIVATE ${OpenBLAS_INCLUDE_DIRS})
