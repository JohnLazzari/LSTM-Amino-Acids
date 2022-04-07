from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from LSTM_train import LSTM

def push_back(list_, index):

    new_list = torch.zeros([1, 100, 20])
    new_list[0][98][index] += 1
    for i in range(list_.shape[1]-2):
        new_list[0][i] = list_[0][i+1]
    
    return new_list

device = 'cuda:0'
val_set = np.load('val.npy', allow_pickle=True)
end_token = torch.zeros([1, 20])

model = LSTM(20, 60, 4).to(device)
model.load_state_dict(torch.load('lstm.pth'))
lr = .001
epochs = 100
criterion = nn.CrossEntropyLoss()

sequence_length = 100
sequences = torch.empty([5, 100, 20])

with torch.no_grad():
    for i in range(5):
        # Get first random sequence
        seed = np.random.randint(0, 19)
        one_hot_amino = torch.zeros([1, 100, 20])
        one_hot_amino[0][98][seed] += 1
        cur_sequence_torch = one_hot_amino

        for j in reversed(range(sequence_length)):
            cur_sequence_torch = cur_sequence_torch.to(device)
            prediction = model(cur_sequence_torch)
            prediction = prediction[-1:]
            prediction = nn.functional.softmax(prediction)
            pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cur_sequence_torch = push_back(cur_sequence_torch, pred)
            
        sequences[i] = cur_sequence_torch[0]

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    encodings = np.eye(len(amino_acids))

    amino_acid_encodings = {}
    for i, amino_acid in enumerate(amino_acids):
        amino_acid_encodings[i] = amino_acid

    print(amino_acid_encodings)

    new_sequences = []
    for i in range(5):
        amino_acid = ''
        for j in range(99):
            character = (sequences[i][j] == 1).nonzero(as_tuple=True)[0]
            amino_acid += amino_acid_encodings[character.item()]
        print(amino_acid)
        new_sequences.append(amino_acid)
