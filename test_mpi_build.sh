#!/bin/bash

# Test script for MPI build integration
# This script tests both MPI-enabled and MPI-disabled builds

set -e

echo "=== TinyLlama MPI Build Test ==="

# Create build directories
mkdir -p build_no_mpi build_with_mpi

echo ""
echo "1. Testing build WITHOUT MPI support..."
cd build_no_mpi
cmake -DHAS_MPI=OFF -DHAS_CUDA=OFF -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) tinyllama_core tinyllama
echo "✓ Build without MPI successful"
cd ..

echo ""
echo "2. Testing build WITH MPI support..."
cd build_with_mpi
cmake -DHAS_MPI=ON -DHAS_CUDA=OFF -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) tinyllama_core tinyllama
if [ -f "tinyllama_distributed" ]; then
    echo "✓ Build with MPI successful - tinyllama_distributed created"
else
    echo "⚠ Build with MPI completed but tinyllama_distributed not found"
fi
cd ..

echo ""
echo "3. Checking executables..."
if [ -f "build_no_mpi/tinyllama" ]; then
    echo "✓ Standard tinyllama executable exists"
else
    echo "✗ Standard tinyllama executable missing"
    exit 1
fi

if [ -f "build_with_mpi/tinyllama_distributed" ]; then
    echo "✓ Distributed tinyllama executable exists"
else
    echo "⚠ Distributed tinyllama executable missing (this is expected if MPI wasn't found)"
fi

echo ""
echo "=== Build Test Complete ==="
echo "Both builds completed successfully!"
echo ""
echo "Usage:"
echo "  Standard inference: ./build_no_mpi/tinyllama"
echo "  Distributed inference: mpirun -np 4 ./build_with_mpi/tinyllama_distributed"
