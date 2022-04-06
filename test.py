from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from LSTM_train import LSTM

device = 'cuda:0'
val_set = np.load('val.npy', allow_pickle=True)
end_token = torch.zeros([1, 20])

model = LSTM(20, 60, 4).to(device)
model.load_state_dict(torch.load('lstm.pth'))
lr = .001
epochs = 100
criterion = nn.CrossEntropyLoss()

target_val = []
val_inp = []

val_labels = []
val_sequences = []

val_batch_size = 793 # prime factor, easier
with torch.no_grad():
    val_acc = 0
    for i in range(len(val_set)+1):

        # create the batches and convert to tensors
        if i % val_batch_size == 0 and i != 0:
            val_labels.append(torch.cat(target_val, dim=0))
            val_sequences.append(torch.stack(val_inp, dim=0))
            # reset
            target_val = []
            val_inp = []

        if i < len(val_set):
            # Get the one hot of the second to last character (before end character)
            one_hot_target_val = torch.Tensor(np.array(val_set[i][-2]))
            # Get it in the form of an index for cross entropy
            target_val.append((one_hot_target_val == 1).nonzero(as_tuple=True)[0])

            # Get the last 51 sequence elements
            cur_sequence_val = torch.Tensor(np.array(val_set[i][-101:]))
            # remove the last 2 elements
            cur_sequence_val = torch.Tensor(cur_sequence_val[:-2])
            # concatenate the remaining sequence with the end token
            cur_sequence_val = torch.cat((cur_sequence_val, end_token), dim=0)

            # If the sequence is not 50 elements, pad it with zeros in the beginning
            while cur_sequence_val.shape[0] < 100:
                cur_sequence_val = torch.cat((end_token, cur_sequence_val), dim=0)

            # append to batch list
            val_inp.append(cur_sequence_val)

    for i, batch in enumerate(val_sequences):

        batch, val_labels[i] = batch.to(device), val_labels[i].to(device)

        prediction = model(batch)
        prediction = prediction[-val_batch_size:]
        loss_val = criterion(prediction, val_labels[i])
        print("Validation Loss: {}".format(loss_val))

        # Getting predictions and training accuracy
        pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        val_acc += pred.eq(val_labels[i].view_as(pred)).sum().item()

    val_acc = 100 * val_acc / (len(val_sequences) * val_batch_size)
    print("Validation Accuracy: {}".format(val_acc))