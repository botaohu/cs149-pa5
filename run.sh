
if [[ `uname` == 'Linux' ]]; then
  export CUDA_HOME=/usr/local/cuda
  export CUDA_LIB=$CUDA_HOME/lib64
elif [[ `uname` == 'Darwin' ]]; then
  export CUDA_HOME=/Developer/NVIDIA/CUDA-5.0
  export CUDA_LIB=$CUDA_HOME/lib
fi 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_LIB:$LD_LIBRARY_PATH


