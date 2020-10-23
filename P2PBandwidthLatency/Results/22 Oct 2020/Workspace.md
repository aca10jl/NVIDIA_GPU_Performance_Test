# Workspace of RTX 3090
- Personal desktop
- CUDA: Release 11.1, V11.1.74
- CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz (Stepping 7)
- GPU: Nvidia RTX 3090 Founders Edition x 2 (NVLinked)
- Motherboard: MSI MEG X299 CREATION (MS-7C06) v1.0
- Memory: G.Skill 64 GB DDR4-3200 CL14 (F4-3200C14Q-64GTZR)
- Storage: Samsung 970 PRO M.2

Outputs from ```collect_env.py```:
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

# Workspace of Tesla V100
- Cluster managed by SLURM
- CUDA: Release 10.1, V10.1.168
- CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz (Stepping 1)
- GPU: Nvidia Tesla V100 SXM2 16GB x 8 in each node
- Motherboard: Supermicro X10DRi v1.02B0
- Memory: N/A
- Storage: N/A

Outputs from ```collect_env.py```:
```
Collecting environment information...
PyTorch version: 1.6.0
Is debug build: False
CUDA used to build PyTorch: 10.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 18.04.4 LTS (x86_64)
GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Clang version: Could not collect
CMake version: version 3.10.2

Python version: 3.7 (64-bit runtime)
Is CUDA available: True
CUDA runtime version: 10.1.168
GPU models and configuration: 
GPU 0: Tesla V100-SXM2-16GB
GPU 1: Tesla V100-SXM2-16GB
GPU 2: Tesla V100-SXM2-16GB
GPU 3: Tesla V100-SXM2-16GB
GPU 4: Tesla V100-SXM2-16GB
GPU 5: Tesla V100-SXM2-16GB
GPU 6: Tesla V100-SXM2-16GB
GPU 7: Tesla V100-SXM2-16GB

Nvidia driver version: 418.126.02
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.6.0.21
/usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.4
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip] numpy==1.18.1
[pip] torch==1.6.0
[pip] torchvision==0.7.0
[conda] blas                      1.0                         mkl  
[conda] cudatoolkit               10.1.243             h6bb024c_0  
[conda] mkl                       2020.0                      166  
[conda] mkl-service               2.3.0            py37he904b0f_0  
[conda] mkl_fft                   1.0.15           py37ha843d7b_0  
[conda] mkl_random                1.1.0            py37hd6b4f25_0  
[conda] numpy                     1.18.1           py37h4f9e942_0  
[conda] numpy-base                1.18.1           py37hde5b4d6_1  
[conda] pytorch                   1.6.0           py3.7_cuda10.1.243_cudnn7.6.3_0    pytorch
[conda] torchvision               0.7.0                py37_cu101    pytorch
```
