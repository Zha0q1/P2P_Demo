# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

# Re-link certain lib in DLC image
execute_process (
    COMMAND bash -c "ln -s /usr/local/cuda/lib64/libnvrtc.so.*.*.* /usr/local/cuda/lib64/libnvrtc.so"
)

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(SMDDP_P2P_DEMO)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -O3 -march=broadwell")

# Certain libs live under this dir
link_directories(/opt/conda/lib)

# Find required packages
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Include dirs
# The P2P header file is in /opt/conda/include/python3.8/smdistributed-dataparallel/
include_directories(${MPI_INCLUDE_PATH})
include_directories(/opt/conda/include/python3.8/smdistributed-dataparallel/)

set(p2p_demo_src main.cpp)

add_executable(p2p_demo ${p2p_demo_src})

# P2P symbols are in /opt/conda/lib/python3.8/site-packages/smdistributed/dataparallel/lib/libsmddpcpp.so
target_link_libraries(p2p_demo /opt/conda/lib/python3.8/site-packages/smdistributed/dataparallel/lib/libsmddpcpp.so)
target_link_libraries(p2p_demo ${MPI_LIBRARIES})
target_link_libraries(p2p_demo ${TORCH_LIBRARIES})
target_link_libraries(p2p_demo ${CUDA_LIBRARIES})

install(TARGETS p2p_demo DESTINATION bin)
