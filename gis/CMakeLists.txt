cmake_minimum_required(VERSION 3.10)
project(GIS CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-O3 ${CMAKE_CUDA_FLAGS}")

add_library(GIS SHARED gis.cu)
target_link_libraries(GIS cudart)

target_compile_features(GIS PUBLIC cxx_std_11)

set_target_properties(GIS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS GIS  LIBRARY DESTINATION $ENV{FOAM_USER_LIBBIN})
