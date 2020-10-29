from torch.utils import data
import dataset as D
import model as M

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# User-provided
epochs = 4
bytecount = 40

# Derived
bits = bytecount * 8
step = 0

logger = SummaryWriter(log_dir="results/metrics")

dataset = D.generate_dummy_dataset("D:/Datasets/ISCXVPN2016/", size=bytecount)
dataloader = D.utils.DataLoader(dataset, batch_size=200, shuffle=True)

classifier = M.SimpleClassifier(in_channels=bits, hidden_channels=[bits, bits, bits // 2, bits // 4, bits // 8, bits // 16], n_classes=5).cuda()
optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

step = 0
for e in range(epochs):
    for i, (x, l) in enumerate(tqdm(dataloader)):

        y = classifier(x.cuda())
        loss = nn.functional.cross_entropy(y, l.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < 0.94:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6

        logger.add_scalar("CE_Loss", loss.item(), step)

        if step % 500 == 0:
            print(f"Step={step} :: loss={loss.item():02f}")

        step += 1
