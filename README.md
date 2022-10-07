# A Developer’s Guide to the P2P Feature in SageMaker Data Parallelism Package

## Overview
In tensor and pipeline parallelism use case, we often need to send tensors directly from one rank to another. PyTorch’s [distributed package](https://pytorch.org/docs/stable/distributed.html) does not provide an easy and async API for this. SageMaker’s [Distributed Data Parallel Library](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-intro.html) fills in this gap by providing a set of intuitive P2P APIs in C++. In this demo, we will run you through the API specs, dev setup, and the demo itself.

```
void herringP2PSend(int rank, const void* buffer, void* mr, uint64_t length,
          int tensor_id, const void* context, uint32_t imm_data);
          
void herringP2PRecv(int rank, void* buffer, void* mr, uint64_t length, int tensor_id,
          const void* context, uint32_t imm_data);   
void* herringP2PRegisterMemory(void* buff, int buff_len);      
```

`herringP2PSend` and `herringP2PRecv` are the two core APIs we will be using. They are asynchronous in the sense that the caller is not blocked by the completion of communication. They require information about the `buffer` pointer (can be either CPU or GPU) and its `length`. Furthermore, `rank` specifies the other party that you would like to communicate with; data transport will happen if this other rank has a matching call with the `same tensor_id`. `context` is a field that the API caller can fully customize. `mr` refers to the memory registration struct returned by `herringP2PRegisterMemory`. All buffers need to be registered first to be able to be transported. You can ignore 'imm_data' for now.

```
struct CompletionEntry {
  const void *context;
  uint32_t dst_rank;
  uint32_t tensor_id;
  herringP2PMessageErrorCode_t error_code;
};
CompletionEntry herringP2PReadCompletion();
CompletionEntry herringP2PSendCompletion();
```
All communication is async. We use the above struct APIs to query the completion of communication requests. Note that `herringP2PReadCompletion` and `herringP2PSendCompletion` are blocking calls — they will return only when *some* communication request is done. The order that completion entries are returned does not depend on the order the communication requests are queued, but rather the order they are completed.

**Failure to match the send/recv calls or query completion when there is none expected will lead to hangs.**


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

## Run the Demo
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

Please refer to the comments in `main.cpp` for exaplainations of each step.

Please notice that this specific demo program runs communication within only one instance, among the local GPUs. You can change the communication pattern freely and communicate across machines if you have a multi-machine cluster setup. MPI is required to correctly boot up the P2P feature, so always use `mpirun` to launch your distribtued jobs. Lastly but not least, please pay attention to `wrapper.sh`. It helps infer the EFA device name from the rank of a process and map it to env var `SMDATAPARALLEL_DEVICE_NAME`. This env var is needed by P2P to locate the correct network card, aka EFA.

