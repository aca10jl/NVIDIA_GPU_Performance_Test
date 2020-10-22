import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
from torchvision.models import resnet152

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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

dtype = torch.float32
x = torch.randn(32, 3, 224, 224).cuda(0).to(dtype)
y = torch.randint(0, 10, (32,)).cuda(0)
m = model().cuda(0).to(dtype)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(m.parameters(), 1e-4)

# warmup
for _ in range(10):
    m.zero_grad()
    output = m(x)
    loss = criterion(output, y)
    loss.backward()

nb_iters = 100

torch.cuda.synchronize()
t0 = time.time()
for _ in range(nb_iters):
    m.zero_grad()
    output = m(x)
    loss = criterion(output, y)
    loss.backward()

torch.cuda.synchronize()
t1 = time.time()
print('{:.3f}ms per iter'.format((t1 - t0)/nb_iters * 1000.))
