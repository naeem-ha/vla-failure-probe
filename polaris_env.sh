# Source inside WSL for the PolaRiS environment.
export CUDA_HOME=$HOME/cuda-13.0
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"  # RTX 5080 (Blackwell) only, keeps kernel builds fast
export OMNI_KIT_ACCEPT_EULA=YES
export POLARIS_DATA_PATH=$HOME/spais-demo/polaris/PolaRiS-Hub
cd ~/spais-demo/polaris
