from pytorch_dataset import model
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
split = 0.9
log_dir = "logs/metrics"
dataset_dir = "D:/Datasets/ISCXVPN2016/"
model_file = "logs/weights/classifier.pth"
results_file = "logs/performance/scores.csv"

# Constant
NCLASSES = 5

# Derived
bits = bytecount * 8
step = 0

logger = SummaryWriter(log_dir=log_dir)

dataset = D.generate_dummy_dataset(dataset_dir, size=bytecount)
train_size = int(len(dataset) * split)
test_size = len(dataset) - train_size
train_ds, test_ds = data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

dataloader = D.utils.DataLoader(train_ds, batch_size=200, shuffle=True)

classifier = M.SimpleClassifier(in_channels=bits, hidden_channels=[bits, bits, bits // 2, bits // 4, bits // 8, bits // 16], n_classes=NCLASSES).cuda()
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

torch.save(classifier.state_dict(), model_file)

results = open(results_file, "w+")

line = [
    "GT",
    *[f"top_{i}" for i in range(NCLASSES)],
    *[f"prob_{i}" for i in range(NCLASSES)]
]
results.write(",".join(line) + "\n")


dataloader = D.utils.DataLoader(test_ds, batch_size=1, shuffle=True)
for i, (x, l) in enumerate(tqdm(dataloader)):
    probs = classifier(x.cuda()).squeeze(0)
    topk = torch.topk(probs, k=NCLASSES)[1]
    results.write(",".join([
        str(l.item()),
        *[str(i.item()) for i in topk],
        *[str(i.item()) for i in probs],
    ]) + "\n")

results.close()
