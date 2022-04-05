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
        h_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to('cuda:0') #hidden state
        c_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to('cuda:0') #internal state
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

targets = torch.empty([len(train_set), 20])
training_inputs = []
end_token = torch.zeros([1, 20])
# quick test for targets
for i in range(len(train_set)):
    targets[i] = torch.Tensor(np.array(train_set[i][-2]))
    new_train = torch.Tensor(np.array(train_set[i][:-2]))
    new_train = torch.cat((new_train, end_token), dim=0)
    training_inputs.append(new_train)

print(targets.shape)
print(training_inputs[0].shape)

model = LSTM(20, 20, 1).to(device)
lr = .001
epochs = 25
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    #running_loss = 0
    for i, sequence in enumerate(training_inputs):

        sequence = sequence.to(device)
        sequence = torch.unsqueeze(sequence, 0)
        target = (targets[i] == 1).nonzero(as_tuple=True)[0]
        target = target.to(device)

        optimizer.zero_grad()
        prediction = model(sequence)
        loss = criterion(prediction, target)
        print(loss)
        #running_loss += loss.item()
        loss.backward()
        optimizer.step()

    #running_loss /= len(training_inputs)
    #print("Epoch {} training loss: {}".format(epoch, running_loss))
        




