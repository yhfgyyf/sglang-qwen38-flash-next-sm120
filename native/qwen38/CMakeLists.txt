cmake_minimum_required(VERSION 3.24)
project(qwen38_native LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
set(PLE_STORE_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/ple_store/target/release/libq38_ple_store.so" CACHE FILEPATH "Rust PLE C ABI library")
if(NOT EXISTS "${PLE_STORE_LIBRARY}")
  message(FATAL_ERROR "Build ple_store first, or set PLE_STORE_LIBRARY")
endif()
add_library(qwen38_native SHARED executor.cpp fp8_decode.cu)
set_target_properties(qwen38_native PROPERTIES CUDA_ARCHITECTURES 120 POSITION_INDEPENDENT_CODE ON)
target_compile_options(qwen38_native PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
# The serving conda environment can load an older libstdc++ before us. No C++
# objects cross this C ABI, so keep our compiler runtime private instead of
# upgrading the environment or depending on import-order-sensitive symbols.
target_link_options(qwen38_native PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
target_link_libraries(qwen38_native PRIVATE CUDA::cudart Threads::Threads "${PLE_STORE_LIBRARY}")

# Host-backed ordinary BF16 attention KV is intentionally a separate C ABI
# library: it has no PLE-store dependency and does not change executor flags or
# lifetime.  The Python owner loads only the library it opts into.
add_library(q38_host_kv SHARED host_kv.cpp host_kv.cu)
set_target_properties(q38_host_kv PROPERTIES CUDA_ARCHITECTURES 120 POSITION_INDEPENDENT_CODE ON)
target_compile_options(q38_host_kv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
target_link_options(q38_host_kv PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
target_link_libraries(q38_host_kv PRIVATE CUDA::cudart)
