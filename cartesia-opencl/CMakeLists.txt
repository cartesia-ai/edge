cmake_minimum_required(VERSION 3.18)

project(cartesia_opencl LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(OpenCL REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ----------------------------- Extensions -----------------------------

# Add library
add_library(cartesia_opencl_ext)

# Add sources
target_sources(
  cartesia_opencl_ext
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/src/ssm_update.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/ssd_update.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/conv1d_forward.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/conv1d_update.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/conv1d_swish_forward.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/conv1d_swish_update.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/ssd_update_no_z.cpp
)

# Add include headers
target_include_directories(
  cartesia_opencl_ext PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}
  ${OpenCL_INCLUDE_DIRS}
)

# Link to OpenCL
target_link_libraries(cartesia_opencl_ext PUBLIC ${OpenCL_LIBRARIES})

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC STABLE_ABI LTO NOMINSIZE
  NB_DOMAIN cartesia_opencl 
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
)
target_link_libraries(_ext PRIVATE cartesia_opencl_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()

# ----------------------------- Installation -----------------------------
install(TARGETS _ext
  LIBRARY DESTINATION cartesia_opencl
  RUNTIME DESTINATION cartesia_opencl
)


