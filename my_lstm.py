import torch
import torch.nn as nn
import torch.functional as F


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTM, self).__init__()
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

        self.Rz = nn.Linear(hidden_size, hidden_size)
        self.Ri = nn.Linear(hidden_size, hidden_size)
        self.Rf = nn.Linear(hidden_size, hidden_size)
        self.Ro = nn.Linear(hidden_size, hidden_size)

        self.W = nn.Linear(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size, input_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        h = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        c = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hs = list()

        for i in range(seq_len):
            xt = x[:, i, :]
            z0 = self.Wz(xt) + self.Rz(h)
            z1 = torch.tanh(z0)
            i0 = self.Wi(xt) + self.Ri(h)
            i1 = torch.sigmoid(i0)
            f0 = self.Wf(xt) + self.Rf(h)
            f1 = torch.sigmoid(f0)
            c = z1 * i1 + f1 * c
            o0 = self.Wo(xt) + self.Ro(h)
            o1 = torch.sigmoid(o0)

            h = o1 * torch.tanh(c)
            hs.append(h.view(batch_size, 1, self.hidden_size))

        h = torch.cat(hs, dim=1)
        h = h[:, seq_len // 2:]
        return self.linear(h)
