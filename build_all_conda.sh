#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set CUDA architecture for your GPU (e.g., compute_89 for NVIDIA Ada GPUs)
export TORCH_CUDA_ARCH_LIST="8.9"

# Install dependencies
pip install torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install -r requirements.txt

# Clone source repository of FoundationPose
git clone https://github.com/NVlabs/FoundationPose.git || echo "FoundationPose already cloned."

# Copy weights and configuration files to the correct directory
mkdir -p FoundationPose/weights/2023-10-28-18-33-37
mkdir -p FoundationPose/weights/2024-01-11-20-02-45

cp "/home/jscheidegger/Documents/New Try/FoundationPoseROS2/Configuration/model_best.pth" FoundationPose/weights/2023-10-28-18-33-37/
cp "/home/jscheidegger/Documents/New Try/FoundationPoseROS2/Configuration/config.yml" FoundationPose/weights/2023-10-28-18-33-37/

# Install pybind11
if [ ! -d "${PROJ_ROOT}/FoundationPose/pybind11" ]; then
    cd "${PROJ_ROOT}/FoundationPose" && git clone https://github.com/pybind/pybind11 && \
        cd pybind11 && git checkout v2.10.0 && \
        mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
        sudo make -j6 && sudo make install
fi

# Install Eigen
if [ ! -d "${PROJ_ROOT}/FoundationPose/eigen-3.4.0" ]; then
    cd "${PROJ_ROOT}/FoundationPose" && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
        tar xvzf ./eigen-3.4.0.tar.gz && rm ./eigen-3.4.0.tar.gz && \
        cd eigen-3.4.0 && \
        mkdir build && \
        cd build && \
        cmake .. && \
        sudo make install
fi

# Clone and install nvdiffrast
if [ ! -d "${PROJ_ROOT}/FoundationPose/nvdiffrast" ]; then
    cd "${PROJ_ROOT}/FoundationPose" && git clone https://github.com/NVlabs/nvdiffrast && \
        cd nvdiffrast && pip install .
fi

# Install mycpp
cd "${PROJ_ROOT}/FoundationPose/mycpp/" && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
sudo make -j$(nproc)

# Install mycuda
cd "${PROJ_ROOT}/FoundationPose/bundlesdf/mycuda" && \
rm -rf build *egg* *.so && \
python3 -m pip install .

cd "${PROJ_ROOT}"
