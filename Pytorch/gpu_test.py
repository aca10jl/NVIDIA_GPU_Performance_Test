import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet152
import time
import argparse

# Hyper-parameters
parser = argparse.ArgumentParser(description='PyTorch GPU Test')
parser.add_argument('--batch_size', '-b', default=32, type=int, help='Batch size')
parser.add_argument('--iteration', '-i', default=100, type=int, help='Test iterations')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID')
parser.add_argument('--seed', '-s', default=666, type=int, help='Random seed')
parser.add_argument('--precision', '-p', default='fp32', choices=['fp32', 'fp16', 'tf32', 'mixed'], type=str.lower, help='Precisions: fp32, fp16, tf32, mixed')
parser.add_argument('--data_parallel', '-dp', action='store_true', help='Data parallelisation')
args = parser.parse_args()

# Set precisions
torch.backends.cudnn.benchmark = True
if args.precision == 'fp32':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# PyTorch random number generator
torch.manual_seed(args.seed)

# Define a neural network
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.resnet = resnet152()
        self.linear = nn.Sequential(nn.Linear(1000, 250),
                                    nn.Linear(250, 64),
                                    nn.Linear(64, 32),
                                    nn.Linear(32, 10),
                                    nn.ReLU())

    def forward(self, x):
        out = self.resnet(x)
        out = self.linear(out)

        return out

# Setup the model
m = model()
if args.data_parallel:
    m = torch.nn.DataParallel(m, device_ids=range(torch.cuda.device_count()))
    args.gpu = None

# Generate random training data
y = torch.randint(0, 10, (args.batch_size,)).cuda(args.gpu)
if args.precision in ['tf32', 'mixed']:
    x = torch.randn(args.batch_size, 3, 224, 224).cuda(args.gpu)
    m = m.cuda(args.gpu)
else:
    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'fp32':
        dtype = torch.float32
    x = torch.randn(args.batch_size, 3, 224, 224).cuda(args.gpu).to(dtype)
    m = m.cuda(args.gpu).to(dtype)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(m.parameters(), 1e-4)
if args.precision == 'mixed':
    scaler = GradScaler()

# Warm up
if args.precision == 'mixed':
    for _ in range(10):
        m.zero_grad()
        with autocast():
            output = m(x)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
else:
    for _ in range(10):
        m.zero_grad()
        output = m(x)
        loss = criterion(output, y)
        loss.backward()

# Main tests start here
torch.cuda.synchronize()
if args.precision == 'mixed':
    t0 = time.time()
    for _ in range(args.iteration):
        m.zero_grad()
        with autocast():
            output = m(x)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
else:
    t0 = time.time()
    for _ in range(args.iteration):
        m.zero_grad()
        output = m(x)
        loss = criterion(output, y)
        loss.backward()
torch.cuda.synchronize()
t1 = time.time()
if args.data_parallel:
    for g in range(torch.cuda.device_count()):
        print('GPU {}:\t{}'.format(g, torch.cuda.get_device_name(g)))
else:
    print('GPU {}:\t{}'.format(args.gpu, torch.cuda.get_device_name(args.gpu)))
print('{}:\t{:.3f}ms per iter'.format(args.precision.upper(), (t1 - t0)/args.iteration * 1000.))
