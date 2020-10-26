from torch.utils import data
import dataset as D
import model as M

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# User-provided
epochs = 10
bytecount = 40

# Derived
bits = bytecount * 8
step = 0

logger = SummaryWriter(log_dir="results/metrics")

dataset = D.generate_dummy_dataset("D:/Datasets/ISCXVPN2016/", size=bytecount)
dataloader = D.utils.DataLoader(dataset, batch_size=10, shuffle=True)

classifier = M.SimpleClassifier(in_features=bits, nclasses=5)
optimizer = optim.Adam(classifier.parameters())

for e in range(epochs):
    for i, (x, l) in enumerate(tqdm(dataloader)):
        step = e * len(dataloader) + i

        y = classifier(x)
        loss = nn.functional.cross_entropy(y, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.add_scalar("CE_Loss", loss.item(), step)

        if step % 500 == 0:
            print(f"Step={step} :: loss={loss.item():02f}")
