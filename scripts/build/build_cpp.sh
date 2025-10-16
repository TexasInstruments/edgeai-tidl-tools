#!/bin/bash

# Default values
BUILD_TYPE="Release"
BUILD_WRAPPER=true
BUILD_EXAMPLES=true
CLEAN_ONLY=false
JOBS=4

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build C++ runtimes wrappers and examples."
    echo ""
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  -c, --clean             Clean build directories without building"
    echo "  -t, --type TYPE         Build type: Debug or Release (default: Release)"
    echo "  -j, --jobs N            Number of parallel jobs for make (default: 4)"
    echo "  --wrapper-only          Build only the wrapper libraries"
    echo "  --examples-only         Build only the examples"
    echo ""
    echo "For more detailed information, see the README.md in this directory."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            usage
            ;;
        -c|--clean)
            CLEAN_ONLY=true
            shift
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --wrapper-only)
            BUILD_WRAPPER=true
            BUILD_EXAMPLES=false
            shift
            ;;
        --examples-only)
            BUILD_WRAPPER=false
            BUILD_EXAMPLES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate build type
if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
    echo "Error: Build type must be either 'Debug' or 'Release'"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the root directory of the project
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Define build directories
WRAPPER_BUILD_DIR="$ROOT_DIR/runtimes/tidl_wrapper/cpp/build"
WRAPPER_BIN_DIR="$ROOT_DIR/runtimes/tidl_wrapper/cpp/bin"
WRAPPER_LIB_DIR="$ROOT_DIR/runtimes/tidl_wrapper/cpp/lib"
EXAMPLES_BUILD_DIR="$ROOT_DIR/runtimes/examples/cpp/build"
EXAMPLES_BIN_DIR="$ROOT_DIR/runtimes/examples/cpp/bin"
EXAMPLES_LIB_DIR="$ROOT_DIR/runtimes/examples/cpp/lib"

# Clean function for wrapper
clean_wrapper() {
    echo "Cleaning wrapper build/bin/lib directory..."
    if [ -d "$WRAPPER_BUILD_DIR" ]; then
        rm -rf "$WRAPPER_BUILD_DIR"
    fi
    if [ -d "$WRAPPER_BIN_DIR" ]; then
        rm -rf "$WRAPPER_BIN_DIR"
    fi
    if [ -d "$WRAPPER_LIB_DIR" ]; then
        rm -rf "$WRAPPER_LIB_DIR"
    fi
}

# Clean function for examples
clean_examples() {
    echo "Cleaning examples build/bin/lib directory..."
    if [ -d "$EXAMPLES_BUILD_DIR" ]; then
        rm -rf "$EXAMPLES_BUILD_DIR"
    fi
    if [ -d "$EXAMPLES_BIN_DIR" ]; then
        rm -rf "$EXAMPLES_BIN_DIR"
    fi
    if [ -d "$EXAMPLES_LIB_DIR" ]; then
        rm -rf "$EXAMPLES_LIB_DIR"
    fi
}

# Build function for wrapper
build_wrapper() {
    echo "Building wrapper libraries ($BUILD_TYPE)..."
    mkdir -p "$WRAPPER_BUILD_DIR"
    cd "$WRAPPER_BUILD_DIR"
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
    make -j$JOBS
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build wrapper libraries"
        exit 1
    fi
    cd "$ROOT_DIR"
    echo "Wrapper libraries built successfully"
}

# Build function for examples
build_examples() {
    echo "Building examples ($BUILD_TYPE)..."
    mkdir -p "$EXAMPLES_BUILD_DIR"
    cd "$EXAMPLES_BUILD_DIR"
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
    make -j$JOBS
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build examples"
        exit 1
    fi
    cd "$ROOT_DIR"
    echo "Examples built successfully"
}

# Main execution
echo "EdgeAI TIDL Tools C++ Build Script"
echo "=================================="

# Handle clean operations
if $BUILD_WRAPPER; then
    clean_wrapper
fi

if $BUILD_EXAMPLES; then
    clean_examples
fi

# Exit if clean only
if $CLEAN_ONLY; then
    echo "Clean operation completed"
    exit 0
fi

# Build wrapper if requested
if $BUILD_WRAPPER; then
    build_wrapper
fi

# Build examples if requested
if $BUILD_EXAMPLES; then
    if $BUILD_WRAPPER; then
        echo "Wrapper libraries built, now building examples..."
    fi
    build_examples
fi

echo "Build process completed successfully"
