import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class ModelFF(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(ModelFF, self).__init__()

        self.input_dims = n_in
        self.hidden_dims = n_hidden
        self.output_dims = n_out

        self.fc1 = nn.Linear(self.input_dims, self.hidden_dims)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dims, self.output_dims)

    def forward(self, x):
        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        return self.fc3(x)

class ModelLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, batch_size):
        super(ModelLSTM, self).__init__()

        self.input_dims = n_in
        self.hidden_dims = n_hidden
        self.output_dims = n_out
        self.batch_size = batch_size
        self.num_layers = 4

        self.hidden = (torch.zeros(1, 1, self.hidden_dims).to('cuda'),
                torch.zeros(1, 1, self.hidden_dims).to('cuda'))

        self.lstm = nn.LSTM(self.input_dims, self.hidden_dims)
        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims*2)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dims*2, self.hidden_dims)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dims, self.output_dims)

    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_dims).to('cuda'),
                torch.zeros(1, 1, self.hidden_dims).to('cuda'))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1), (self.hidden[0].detach(), self.hidden[1].detach()))
        x = lstm_out.view(len(x), -1)
        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        return self.fc3(x)