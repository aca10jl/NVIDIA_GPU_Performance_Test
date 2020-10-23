import torch
import time
from torchvision.models import resnet152

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda:0')
b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda:0')
ab_full = a_full @ b_full
mean = ab_full.abs().mean()  # 80.7277

a = a_full.float()
b = b_full.float()

# warmup
for _ in range(5):
	# Do matmul at TF32 mode.
	ab_tf32 = a @ b  # takes 0.016s on GA100
	error = (ab_tf32 - ab_full).abs().max()  # 0.1747
	relative_error = error / mean  # 0.0022

	# Do matmul with TF32 disabled.
	torch.backends.cuda.matmul.allow_tf32 = False
	ab_fp32 = a @ b  # takes 0.11s on GA100
	error = (ab_fp32 - ab_full).abs().max()  # 0.0031
	relative_error = error / mean  # 0.000039

# test
nb_iters = 100

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.synchronize()
t0 = time.time()
for _ in range(nb_iters):
    ab_tf32 = a @ b

torch.cuda.synchronize()
t1 = time.time()
print('TF32: {:.3f}ms per iter'.format((t1 - t0)/nb_iters * 1000.))

torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.synchronize()
t0 = time.time()
for _ in range(nb_iters):
    ab_fp32 = a @ b

torch.cuda.synchronize()
t1 = time.time()
print('FP32: {:.3f}ms per iter'.format((t1 - t0)/nb_iters * 1000.))


