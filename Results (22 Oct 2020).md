# 22 Oct 2020
```
Test with GPU 0, where there is no monitor cables plugged.
Fan speed is controlled automatically by the GPU BIOS.
Record the best result from three runs (e.g. python fp16.py && python fp16.py && python fp16.py).
```
## fp16
103.245ms per iter

## fp32
196.978ms per iter

## tf32
189.300ms per iter

## matmul
TF32: 57.170ms per iter

FP32: 101.313ms per iter

## mixed precision
125.989ms per iter

## Workspace
```
PyTorch version: 1.8.0a0+13decdd
Is debug build: True
CUDA used to build PyTorch: 11.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.1 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.18.2

Python version: 3.8 (64-bit runtime)
Is CUDA available: True
CUDA runtime version: Could not collect
GPU models and configuration: 
GPU 0: GeForce RTX 3090
GPU 1: GeForce RTX 3090

Nvidia driver version: 455.23.05
cuDNN version: Probably one of the following:
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.1
[pip3] torch==1.8.0a0
[pip3] torchvision==0.8.0a0+3756b60
[conda] blas                      1.0                         mkl  
[conda] mkl                       2020.2                      256  
[conda] mkl-include               2020.2                      256  
[conda] mkl-service               2.3.0            py38he904b0f_0  
[conda] mkl_fft                   1.2.0            py38h23d657b_0  
[conda] mkl_random                1.1.1            py38h0573a6f_0  
[conda] numpy                     1.19.1           py38hbc911f0_0  
[conda] numpy-base                1.19.1           py38hfa32c7d_0  
[conda] torch                     1.8.0a0                  pypi_0    pypi
[conda] torchvision               0.8.0a0+3756b60          pypi_0    pypi
```
