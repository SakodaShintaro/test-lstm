import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, out.shape[1] // 2:]
        return self.linear(out)
