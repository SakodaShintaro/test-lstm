from my_lstm import MyLSTM
import torch
from lstm import LSTM
import time

MAX_STEP = 50000
BATCH_SIZE = 256
N = 11
M = 12

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"device = {device}")

hidden_size = 128
model = LSTM(M, hidden_size)
# model = MyLSTM(M, hidden_size)
model.to(device)
crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

start = time.time()

for i in range(1, MAX_STEP + 1):
    x = torch.randint(1, M, (BATCH_SIZE, N)).to(device)
    y = torch.zeros((BATCH_SIZE, N)).to(torch.int64).to(device)
    input_x = torch.cat([x, y], dim=1).to(device)
    input_x = torch.eye(M)[input_x].to(device)
    logit = model.forward(input_x)
    logit = logit.view(-1, M)
    teacher = x.view(-1)

    loss = crit(logit, teacher)

    # 正答率の計算
    pred = torch.argmax(logit, dim=1)
    compare = (pred == teacher)
    accuracy_all = (compare.sum().float() / compare.numel())
    compare_batch = compare.view([BATCH_SIZE, N]).all(dim=1)
    accuracy_batch = (compare_batch.sum().float() / BATCH_SIZE)

    elapsed = time.time() - start
    print(f"{elapsed:5.1f}\t{i:4d}\t{loss.item():.4f}\t{accuracy_all.item():.4f}\t{accuracy_batch.item():.4f}", end="\r")

    if i % (MAX_STEP // 25) == 0:
        print()

    optim.zero_grad()
    loss.backward()
    optim.step()
