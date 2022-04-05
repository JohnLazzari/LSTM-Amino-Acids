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

batch_size = 256
batch_targets = []
batch_inputs = []
# quick way to implement batches, should do this better
for i in range(len(train_set)):

    # create the batches and convert to tensors
    if i % batch_size == 0 and i != 0:
        targets.append(torch.cat(batch_targets, dim=0))
        training_inputs.append(torch.stack(batch_inputs, dim=0))
        # reset
        batch_targets = []
        batch_inputs = []

    # Get the one hot of the second to last character (before end character)
    one_hot_target = torch.Tensor(np.array(train_set[i][-2]))
    # Get it in the form of an index for cross entropy
    batch_targets.append((one_hot_target == 1).nonzero(as_tuple=True)[0])

    # Get the last 51 sequence elements
    cur_sequence = torch.Tensor(np.array(train_set[i][-51:]))
    # remove the last 2 elements
    cur_sequence = torch.Tensor(cur_sequence[:-2])
    # concatenate the remaining sequence with the end token
    cur_sequence = torch.cat((cur_sequence, end_token), dim=0)

    # If the sequence is not 50 elements, pad it with zeros in the beginning
    while cur_sequence.shape[0] < 50:
        cur_sequence = torch.cat((end_token, cur_sequence), dim=0)

    # append to batch list
    batch_inputs.append(cur_sequence)

# Standard training components
model = LSTM(20, 60, 4).to(device)
lr = .001
epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # keep track of epoch loss
    running_loss = 0
    correct = 0
    for i, sequence in enumerate(training_inputs):
        # Put data to devices
        sequence, targets[i] = sequence.to(device), targets[i].to(device)

        optimizer.zero_grad()
        prediction = model(sequence)
        # Get the final hidden state for the batch, should be the last 256
        prediction = prediction[-256:]
        loss = criterion(prediction, targets[i])

        # Getting predictions and training accuracy
        pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(targets[i].view_as(pred)).sum().item()

        # keep track of loss
        running_loss += loss.item()
        # Update parameters
        loss.backward()
        optimizer.step()

    running_loss /= len(training_inputs)
    print("Epoch {} training loss: {}, Training acc: {}".format(epoch, running_loss, 100 * correct / (len(training_inputs)*256)))