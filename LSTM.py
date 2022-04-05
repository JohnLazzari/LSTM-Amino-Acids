from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to('cuda:0') #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to('cuda:0') #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

device = 'cuda:0'
train_set = np.load('train.npy', allow_pickle=True)
val_set = np.load('val.npy', allow_pickle=True)
print(np.shape(train_set))

targets = []
training_inputs = []
end_token = torch.zeros([1, 20])
# quick test for targets
for i in range(len(train_set)):
    one_hot_target = torch.Tensor(np.array(train_set[i][-2]))
    targets.append((one_hot_target == 1).nonzero(as_tuple=True)[0])
    new_train = torch.Tensor(np.array(train_set[i][:-2]))
    new_train = torch.cat((new_train, end_token), dim=0)
    new_train = torch.unsqueeze(new_train, dim=0)
    training_inputs.append(new_train)

model = LSTM(20, 40, 1).to(device)
lr = .001
epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    running_loss = 0
    for i, sequence in enumerate(training_inputs):
        # Put data to devices
        sequence, targets[i] = sequence.to(device), targets[i].to(device)

        # Standard training protocol, takes very long too much data
        # Might need to implement batches, maybe add layers or make bidirectional
        optimizer.zero_grad()
        prediction = model(sequence)
        loss = criterion(prediction, targets[i])
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    running_loss /= len(training_inputs)
    print("Epoch {} training loss: {}".format(epoch, running_loss))