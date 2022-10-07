// Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
//
// http://aws.amazon.com/apache2.0/
//
// or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and limitations under the License.

#include <iostream>
#include <mpi.h>
#include <herring.hpp>
#include <cuda_runtime.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    fprintf(stderr,                                 \
        "SMDDPCUDAError: CUDA error %s:%d '%s'\n",  \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    MPI_Abort(MPI_COMM_WORLD, 1);                   \
  }                                                 \
} while(0) 

void create_and_register_buffer(void*& buf, size_t length, void*& mr) {
    // Allocate GPU buffer
    CUDACHECK(cudaMallocHost(&buf, length));
    // Register the buffer with RDMA so that communication is enabled
    // mr stand for memory registration
    mr = herringP2PRegisterMemory(buf, length);
}

int main() {
    // MPI is required
    int world_size, world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "Hello from P2P Demo" << std::endl;
    std::cout << "world_size " << world_size << " world_rank " << world_rank << std::endl;

    // Initialization is quired to use P2P
    herringP2PInitialize();

    // API to get local device index
    std::cout << "Local Device (GPU Index): " << std::endl;
    std::cout << herringLocalDevice() << std::endl;

    // Set device
    CUDACHECK(cudaSetDevice(herringLocalDevice()));

    std::vector<std::string> devices;
    herringP2PGetRDMADevices(devices);
    std::cout << "RDMA Devices:" << std::endl;
    for (auto& device : devices) {
        std::cout << device << std::endl;
    }

    //---- Create GPU buffer and register it with RDMA
    size_t length = 1024;
    void* buf, *mr;
    create_and_register_buffer(buf, length, mr);

    //---- Populate the GPU buffer
    int data = world_rank;
    CUDACHECK(cudaMemcpy(buf, &data, sizeof(int), cudaMemcpyHostToDevice));

    //---- Even ranks send to odd ranks. 0 sends to 1, 2 sends to 3 and so on.
    // The communicaiton pattern can be changed freely.
    // Any rank can send to/recv from everyone else. Communication can happen both intra-
    // and inter-node.
    int tensor_id = 9876;
    void* context;
    int rank;
    if (world_rank % 2 == 0) {
        rank = world_rank + 1;
        context = (void *)0x1234;
        std::cout << "Sending..." <<
	    " rank is " << rank <<
            " buf is " << buf <<
            " mr is " << mr <<
            " length is " << length <<
            " tensor_id is " << tensor_id <<
            " context is " << context << std::endl;
        // Send the buffer, this API is async will return immediately
	herringP2PSend(rank, buf, mr, length, tensor_id, context, 0);
	// Pop a completion entry for send. Blocks until any completion entry becomes available
	auto completion_entry = herringP2PSendCompletion();
        std::cout << "Send is done" <<
            " context is " << completion_entry.context << 
            " dst_rank is " << completion_entry.dst_rank <<
            " tensor_id is " << completion_entry.tensor_id <<
            " error_code is " << completion_entry.error_code << std::endl;;
    } else {
        rank = world_rank - 1;
        context = (void *)0x4567;
        std::cout << "Receiving..." <<
            " rank is " << rank <<
            " buf is " << buf <<
            " mr is " << mr <<
            " length is " << length <<
            " tensor_id is " << tensor_id <<
            " context is " << context << std::endl;
        // Receive the buffer, this API is async will return immediately
        herringP2PRecv(rank, buf, mr, length, tensor_id, context, 0);
        // Pop a completion entry for receive. Blocks until any completion entry becomes available
	auto completion_entry = herringP2PReadCompletion();
        std::cout << "Recv is done" <<
            " context is " << completion_entry.context <<
            " dst_rank is " << completion_entry.dst_rank <<
            " tensor_id is " << completion_entry.tensor_id <<
            " error_code is " << completion_entry.error_code << std::endl;
    }

    //---- Check content of GPU buffer
    CUDACHECK(cudaMemcpy(&data, buf, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Rank " << world_rank << " has data " << data << std::endl;

    // Shutdown is required
    herringP2PShutdown();

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
