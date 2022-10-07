# A Developer’s Guide to the P2P Feature in SageMaker Data Parallelism Package

## Overview

## Example DLC Image
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker
```
## C++ Header File
```
/opt/conda/include/python3.8/smdistributed-dataparallel/herring.hpp
```
Only the following structs/APIs are relevant to P2P.
```
// The status of P2P Send/Recv
typedef enum {
  RDMA_READ_SUCCESS = 0,
  RDMA_READ_FAIL,
  RDMA_READ_STATUS_UNKOWN,
  RDMA_SHUTDOWN,
} herringP2PMessageErrorCode_t;

// A completion entry is created for each Send/Recv call. The communication is async,
// so we use completion entries to identify which calls are done
struct CompletionEntry {
  const void *context; // context is a void pointer that the API user can customize
  uint32_t dst_rank; // rank that to which data is sent/from which data is received
  uint32_t tensor_id; // unique id that match Send/Recv in paris
  herringP2PMessageErrorCode_t error_code; // status of the Send/Recv call
};

typedef CompletionEntry* CompletionEntryPtr;

// All buffers to be sent/received need to be registered
// The returned pointer by this API needs to be passed into Send/Recv calls (mr)
void* herringP2PRegisterMemory(void* buff, int buff_len);
  
// Pop one completion entry for receive
// Blocks until any is available
CompletionEntry herringP2PReadCompletion();

// Pop one completion entry for send
// Blocks until any is available
CompletionEntry herringP2PSendCompletion();

// Send
// rank: dst rank
// buffer: buffer to be sent
// mr: memory registration, returned by herringP2PRegisterMemory()
// length: length in bytes to be sent
// tensor_id: the unique id that match Send/Recv in pairs
// context: a void pointer that can be customized
// imm_data: ignore for now, use any value
void herringP2PSend(int rank, const void* buffer, void* mr, uint64_t length,
          int tensor_id, const void* context, uint32_t imm_data);

// Receive
// rank: src rank
// buffer: buffer to receive into
// mr: memory registration, returned by herringP2PRegisterMemory()
// length: length in bytes to be received
// tensor_id: the unique id that match Send/Recv in pairs
// context: a void pointer that can be customized
// imm_data: ignore for now, use any value
void herringP2PRecv(int rank, void* buffer, void* mr, uint64_t length, int tensor_id,
          const void* context, uint32_t imm_data);

// Returns a list of RDMA device names
void herringP2PGetRDMADevices(std::vector<std::string>&);

// Init. Must call before any P2P APIs
void herringP2PInitialize();

// Shutdown. Must call before process exits.
void herringP2PShutdown();

// Returns the index of local device (GPU)
int herringLocalDevice();
```

## Supported Instance Type
```
p4d.24xlarge
```

## Dev Setup

We recommend launchingp4d.24xlarge (https://aws.amazon.com/ec2/instance-types/p4/) instances on EC2 for development. Please notice that each p4d instance comes with 4 EFA (https://aws.amazon.com/hpc/efa/) network cards. Be sure to add enough network interfaces to expose all the EFA devices when you launch the instance. Also, when configuring multiple instances, please be sure to include them in the same subnet to enable inter-node communication.

We also recommend our DLC PyTorch image in a dockerized environment. Be sure to expose the EFA devices accordingly:
```
root@compute-dy-p4d24xlarge-1:/workspace/build# ls /dev/infiniband/
rdma_cm  uverbs0  uverbs1  uverbs2  uverbs3
```
Example docker command:
```
docker run --runtime=nvidia --gpus 8 \
    --privileged \
    --rm \
    -d \
    --name $CONTAINER_NAME \
    --uts=host --ulimit stack=67108864 --ulimit memlock=-1 --ipc=host --net=host \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --security-opt seccomp=unconfined  \
    ...
```
Once you are in the docker container, pull this very GitHub repo. Run `bash pre.sh` to install dev libraries such as CuDNN.

To build and run the demo, do:

```
mkdir build && cd build
export Torch_DIR=/opt/conda/lib/python3.8/site-packages/torch && cmake .. && cmake --build .
```

To run the example, do
```
mpirun -N 8 bash wrapper.sh <inset_correct_dir_here>/p2p_demo
```

Please notice that this specific demo program runs communication within only one instance, among the local GPUs. You can change the communication pattern freely and communicate across machines, if you have a multi-machine cluster setup. MPI is required to correctly boot up the P2P feature, so always use `mpirun` to launch your distribtued jobs. Lastly but not least, please pay attention to `wrapper.sh`. It helps infer the EFA device name from the rank of a process and map it to env var `SMDATAPARALLEL_DEVICE_NAME`. This env var is needed by P2P to locate the correct net work card, aka EFA.

