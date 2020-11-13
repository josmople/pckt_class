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
epochs = 10
bytecount = 40
split = 0.9
log_dir = "logs/metrics"
dataset_dir = "D:/Datasets/ISCXVPN2016/"
model_file = "logs/weights/classifier.pth"
results_file = "logs/performance/scores.csv"
lr_init = 1e-4
lr_epoch_updates = [0.4, 0.8]
seed = 42

train_phase = True
eval_phase = True

# Constant
NCLASSES = 5
ADDRESS_BYTES = 8

# Derived
bits = bytecount * 8
address_bits = ADDRESS_BYTES * 8
usable_bits = bits - address_bits
step = 0


classifier = M.SimpleClassifier(in_channels=usable_bits, hidden_channels=[bits, bits, bits // 2, bits // 4, bits // 8, bits // 16], n_classes=NCLASSES).cuda()
dataset = D.generate_dummy_dataset(dataset_dir, size=bytecount)
train_size = int(len(dataset) * split)
test_size = len(dataset) - train_size
train_ds, test_ds = data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

if train_phase:

    optimizer = optim.Adam(classifier.parameters(), lr=lr_init)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * r) for r in lr_epoch_updates], gamma=0.1)

    step = 0
    loss = 0

    dataloader = D.utils.DataLoader(train_ds, batch_size=1000, shuffle=True)
    logger = SummaryWriter(log_dir=log_dir)
    for e in range(epochs):
        print(f"--- EPOCH {e:02} ---")
        for i, (x, l) in enumerate(tqdm(dataloader)):

            y = classifier(x.cuda())
            loss = nn.functional.cross_entropy(y, l.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("CE_Loss", loss.item(), step)

            if step % 500 == 0:
                print(f"Step={step} :: loss={loss.item():02f}")

            step += 1

        scheduler.step()

    print("Final loss: ", loss.item())

    torch.save(classifier.state_dict(), model_file)

if eval_phase:
    classifier.load_state_dict(torch.load(model_file))

    results = open(results_file, "w+")

    line = [
        "GT",
        *[f"top_{i}" for i in range(NCLASSES)],
        *[f"prob_{i}" for i in range(NCLASSES)]
    ]
    results.write(",".join(line) + "\n")

    correct1 = 0
    correct2 = 0

    dataloader = D.utils.DataLoader(test_ds, batch_size=1, shuffle=True)
    total = len(dataloader)
    with torch.no_grad():
        for i, (x, l) in enumerate(tqdm(dataloader)):
            scores = classifier(x.cuda()).squeeze(0).cpu()
            probs = torch.softmax(scores, dim=0)
            topk = torch.topk(probs, k=NCLASSES)[1]
            results.write(",".join([
                str(l.item()),
                *[str(i.item()) for i in topk],
                *[str(i.item()) for i in probs],
            ]) + "\n")
            if l == topk[0]:
                correct1 += 1
            if l == topk[0] or l == topk[1]:
                correct2 += 1

    print(f"Accuracy top-1: {correct1/total:.4f}, Accuracy top-2: {correct2/total:.4f}")

    results.close()
