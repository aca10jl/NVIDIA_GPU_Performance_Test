import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
from torchvision.models import resnet152

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

x = torch.randn(32, 3, 224, 224).cuda(0)
y = torch.randint(0, 10, (32,)).cuda(0)
m = model().cuda(0)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(m.parameters(), 1e-4)
scaler = GradScaler()

# warmup
for _ in range(10):
    m.zero_grad()
    with autocast():
        output = m(x)
        loss = criterion(output, y)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

nb_iters = 100

torch.cuda.synchronize()
t0 = time.time()
for _ in range(nb_iters):
    m.zero_grad()
    with autocast():
        output = m(x)
        loss = criterion(output, y)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

torch.cuda.synchronize()
t1 = time.time()
print('{:.3f}ms per iter'.format((t1 - t0)/nb_iters * 1000.))



