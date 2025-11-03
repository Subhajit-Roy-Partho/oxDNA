#!/bin/bash

# Build script for DNA2 Energy Calculator
# This script sets up the build environment and compiles the project

set -e  # Exit on any error

echo "DNA2 Energy Calculator Build Script"
echo "==================================="

# Check if PyTorch is installed
echo "Checking for PyTorch installation..."
if ! python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo "Error: PyTorch not found. Please install PyTorch first."
    echo "Visit: https://pytorch.org/get-started/locally/"
    exit 1
fi

# Get PyTorch installation directory
PYTORCH_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)
echo "PyTorch CMake path: $PYTORCH_DIR"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_PREFIX_PATH="$PYTORCH_DIR" ..

# Build the project
echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo ""
echo "Executables:"
echo "  - bin/dna2_example    : Example usage"
echo "  - bin/dna2_test       : Basic tests"
echo ""
echo "To run the example:"
echo "  cd build && ./bin/dna2_example"
echo ""
echo "To run the tests:"
echo "  cd build && ./bin/dna2_test"
echo ""
echo "To install (optional):"
echo "  cd build && sudo make install"