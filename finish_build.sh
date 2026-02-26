#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=192G
#SBATCH -t 02:00:00
#SBATCH -J finish_vllm_build
#SBATCH -o build_output_%j.txt

# 1. Load the same environment you used interactively
ml load CUDA/12.8.0
ml load GCC/13.3.0
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=12
export NVCC_THREADS=2

# 2. These ensure uv finds your existing progress and build tools
export UV_PREFER_BINARY=1
export PYTHONHTTPSVERIFY=0

# 3. Run the install - ninja will see the 301 files and jump to the end
echo "Starting final build phase at $(date)"
uv add vllm --no-build-isolation --offline

# 4. Verification
echo "Verifying installation..."
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
uv run python -c "import vllm; print('vLLM Build Success!')"
