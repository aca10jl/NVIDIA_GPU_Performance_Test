# NVIDIA_GPU_Performance_Test
Test the performance of NVIDIA GPUs.

## Usage
1. Collect details of the workspace:
```bash
python collect_env.py
```
2. Test on FP16
```bash
python fp16.py
```
3. Test on FP32
```bash
python fp32.py
```
4. Test on [TP32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
```bash
python tp32.py
```
5. Test on matmul operations
```bash
python matmul.py
```
6. Test on [Auto Mixed Precision](https://developer.nvidia.com/automatic-mixed-precision)
```bash
python mixed_precision.py
```
7. P2P Bandwidth and Latency Tests
- Compile the application. You may need to modify the makefile ([#1](https://github.com/aca10jl/NVIDIA_GPU_Performance_Test/blob/71f9883fe170e21dcca2de49625232db3717b248/P2PBandwidthLatency/Makefile#L271) and [#2](https://github.com/aca10jl/NVIDIA_GPU_Performance_Test/blob/71f9883fe170e21dcca2de49625232db3717b248/P2PBandwidthLatency/Makefile#L273)) to adapt to your GPU archtechture. 
```bash
make
```
- Test
```bash
./p2pBandwidthLatencyTest
```
