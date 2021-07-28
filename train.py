import torch
from lstm import LSTM

MAX_STEP = 1000
BATCH_SIZE = 16
N = 2
M = 3

input_size = M
hidden_size = 256
model = LSTM(input_size, hidden_size)
crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for i in range(MAX_STEP):
    x = torch.randint(1, M, (BATCH_SIZE, N))
    y = torch.zeros((BATCH_SIZE, N)).to(torch.int64)
    input_x = torch.cat([x, y], dim=1)
    input_x = torch.eye(M)[input_x]
    logit = model.forward(input_x)
    logit = logit.view(-1, M)
    teacher = x.view(-1)

    loss = crit(logit, teacher)
    print(f"{i:4d}\t{loss.item():.4f}")

    optim.zero_grad()
    loss.backward()
    optim.step()
