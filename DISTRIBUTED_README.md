# TinyLlama Distributed Inference

This document describes the distributed inference capabilities of TinyLlama using MPI (Message Passing Interface).

## Overview

The distributed inference system allows you to run TinyLlama across multiple nodes in a cluster, enabling:
- **Horizontal scaling**: Distribute inference requests across multiple machines
- **Load balancing**: Automatically route requests to available nodes
- **High availability**: Continue processing even if some nodes fail
- **Resource optimization**: Utilize GPU resources across multiple machines

## Architecture

### Node Roles

- **Master Node**: Coordinates requests and aggregates results (typically rank 0)
- **Worker Nodes**: Process inference requests using local model instances
- **Hybrid Nodes**: Can act as both master and worker

### Communication

- Uses MPI for inter-node communication
- Automatic cluster discovery and topology mapping
- Heartbeat system for node health monitoring
- Efficient serialization for request/response data

## Building with MPI Support

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler
- CUDA Toolkit (optional, for GPU acceleration)
- MPI implementation (MPICH will be automatically downloaded if not found)

### Build Instructions

```bash
# Configure with MPI support
cmake -DHAS_MPI=ON -DHAS_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j$(nproc)
```

The build system will:
1. Try to find an existing MPI installation
2. If not found, automatically download and build MPICH using FetchContent
3. Create the `tinyllama_distributed` executable

## Usage

### Basic Distributed Setup

```bash
# Run with 4 nodes (1 master + 3 workers)
mpirun -np 4 ./tinyllama_distributed model.gguf tokenizer.model worker -1

# Run with hybrid nodes (all can be master/worker)
mpirun -np 4 ./tinyllama_distributed model.gguf tokenizer.model hybrid 32
```

### Command Line Arguments

```
./tinyllama_distributed <model_path> <tokenizer_path> <role> [gpu_layers]

Arguments:
  model_path     Path to model file (.gguf) or directory (SafeTensors)
  tokenizer_path Path to tokenizer file
  role          Node role: 'master', 'worker', or 'hybrid'
  gpu_layers    Number of GPU layers (-1 for all, 0 for CPU-only)
```

### Interactive Usage

When running as master node, you get an interactive prompt:

```
=== TinyLlama Distributed Inference (Master Node) ===
Cluster info:
  Node 0 (hostname1): MASTER, GPU: yes, Layers: -1
  Node 1 (hostname2): WORKER, GPU: yes, Layers: -1
  Node 2 (hostname3): WORKER, GPU: yes, Layers: -1

tinyllama> What is the capital of France?
Processing request...

--- Response ---
The capital of France is Paris, which is located in the north-central part of the country...

Stats:
  Tokens: 45
  Inference time: 234ms
  Total time: 267ms
  Processed by node: 1

tinyllama> status
Cluster Status:
  Total requests: 1
  Completed: 1
  Failed: 0
  Active requests per node:
    Node 0: 0
    Node 1: 0
    Node 2: 0
```

### Environment Variables

- `TINYLLAMA_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Advanced Configuration

### Cluster Deployment

For production deployments, consider:

1. **Network Configuration**: Ensure all nodes can communicate via MPI
2. **Shared Storage**: Model files should be accessible from all nodes
3. **Load Balancing**: Use hybrid nodes for better resource utilization
4. **Monitoring**: Enable debug logging to monitor cluster health

### Performance Tuning

- **GPU Memory**: Adjust `gpu_layers` based on available VRAM per node
- **Batch Size**: Larger clusters can handle more concurrent requests
- **Network**: Use high-speed interconnects (InfiniBand) for best performance

### Example Deployment Scripts

#### SLURM Cluster
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

mpirun ./tinyllama_distributed model.gguf tokenizer.model hybrid -1
```

#### Docker Swarm
```yaml
version: '3.8'
services:
  tinyllama-distributed:
    image: tinyllama:mpi
    deploy:
      replicas: 4
    command: ["mpirun", "-np", "4", "./tinyllama_distributed", "model.gguf", "tokenizer.model", "hybrid", "-1"]
```

## API Integration

The distributed coordinator can be integrated into existing applications:

```cpp
#include "distributed_coordinator.h"

// Create coordinator
DistributedCoordinator coordinator(
    DistributedCoordinator::NodeRole::MASTER,
    "model.gguf", 
    "tokenizer.model", 
    -1
);

// Initialize cluster
coordinator.initialize();

// Submit request
DistributedCoordinator::InferenceRequest request;
request.request_id = "req_001";
request.prompt = "Hello, world!";
request.max_tokens = 100;

coordinator.submit_request(request, [](const auto& result) {
    std::cout << "Response: " << result.generated_text << std::endl;
});
```

## Troubleshooting

### Common Issues

1. **MPI Initialization Fails**
   - Check MPI installation: `mpirun --version`
   - Verify network connectivity between nodes
   - Ensure firewall allows MPI communication

2. **Model Loading Errors**
   - Verify model files exist on all nodes
   - Check file permissions and paths
   - Ensure sufficient memory/storage

3. **GPU Issues**
   - Verify CUDA installation: `nvidia-smi`
   - Check GPU memory availability
   - Adjust `gpu_layers` parameter

### Debug Mode

Enable detailed logging:
```bash
export TINYLLAMA_LOG_LEVEL=DEBUG
mpirun -np 4 ./tinyllama_distributed model.gguf tokenizer.model worker -1
```

## Limitations

- Model files must be accessible from all nodes (shared filesystem or local copies)
- Currently supports LLaMA-type models only
- Requires homogeneous cluster (same architecture/capabilities)
- No dynamic node addition/removal during runtime

## Future Enhancements

- [ ] Dynamic cluster scaling
- [ ] Model sharding across nodes
- [ ] Heterogeneous cluster support
- [ ] REST API for distributed inference
- [ ] Kubernetes operator
- [ ] Performance monitoring dashboard
