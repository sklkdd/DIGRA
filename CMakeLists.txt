cmake_minimum_required(VERSION 3.22)
project(Digra)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Fixed: Removed -march=native and -mavx512f to avoid "Illegal instruction" crashes in Docker
# Use portable x86-64 with AVX support instead of native CPU-specific optimizations
SET( CMAKE_CXX_FLAGS  "-O3 -march=x86-64 -lrt -DHAVE_CXX0X -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -mavx" )


option(ENABLE_SSE "Enable SSE support" ON)
add_executable(Digra main.cpp hnswlib/bruteforce.h hnswlib/hnswalg.h hnswlib/hnswlib.h hnswlib/space_ip.h hnswlib/space_l2.h hnswlib/stop_condition.h hnswlib/visited_list_pool.h 
        utils.hpp
        DataMaker.hpp
        TreeHNSW.hpp
)

# FANNS benchmarking wrappers
add_executable(csv_to_data apps/csv_to_data_converter.cpp)

add_executable(build_wrapper apps/build_wrapper.cpp
        hnswlib/bruteforce.h hnswlib/hnswalg.h hnswlib/hnswlib.h 
        hnswlib/space_ip.h hnswlib/space_l2.h hnswlib/stop_condition.h 
        hnswlib/visited_list_pool.h
        utils.hpp
        TreeHNSW.hpp
)

add_executable(search_wrapper apps/search_wrapper.cpp
        hnswlib/bruteforce.h hnswlib/hnswalg.h hnswlib/hnswlib.h 
        hnswlib/space_ip.h hnswlib/space_l2.h hnswlib/stop_condition.h 
        hnswlib/visited_list_pool.h
        utils.hpp
        TreeHNSW.hpp
)

# Combined build+query wrapper (like SeRF) - builds index once and queries with multiple ef_search values
add_executable(index_construction_and_query_execution apps/index_construction_and_query_execution.cpp
        hnswlib/bruteforce.h hnswlib/hnswalg.h hnswlib/hnswlib.h 
        hnswlib/space_ip.h hnswlib/space_l2.h hnswlib/stop_condition.h 
        hnswlib/visited_list_pool.h
        utils.hpp
        TreeHNSW.hpp
)
