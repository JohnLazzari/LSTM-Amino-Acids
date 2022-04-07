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
train_set = np.load('train.npy', allow_pickle=True)
end_token = torch.zeros([1, 20])

model = LSTM(20, 60, 4).to(device)
model.load_state_dict(torch.load('lstm.pth'))
lr = .001
epochs = 100
criterion = nn.CrossEntropyLoss()

# Getting the training set 3-gram, go through the training set and get the values of a window of 3 amino acids
# the indices in sequence count correspond to the sequence, so add one if found to the correct index
sequence_count = torch.zeros([20, 20, 20])
for i in range(len(train_set)):
    for j in range(len(train_set[i])-3):
        index_1 = np.argmax(train_set[i][j], axis=0)
        index_2 = np.argmax(train_set[i][j+1], axis=0)
        index_3 = np.argmax(train_set[i][j+2], axis=0)

        sequence_count[index_1][index_2][index_3] += 1

# divide by the training set to get probs
sequence_count /= len(train_set)
sequence_count = sequence_count.flatten()

plt.plot(sequence_count)
plt.show()

# Get the model 3-gram, generate a sequence while taking the probabilities of p(a)p(b|a)p(c|a,b)
model_probs = torch.empty([20, 20, 20])
with torch.no_grad():
    one_hot = torch.zeros([1, 100, 20])
    for i in range(20):
        for j in range(20):
            for k in range(20):
                # first input, all zeros to get prob of first term (ex. A)
                one_hot = torch.zeros([1, 100, 20]).to(device)
                pred = model(one_hot)
                pred = pred[-1]
                pred = nn.functional.softmax(pred)
                first_prob = pred[i]

                # second input, add the current term to the sequence to generate next
                # probability for index j, ex. (A, [end])
                one_hot[0][98][i] += 1
                pred = model(one_hot)
                pred = pred[-1]
                pred = nn.functional.softmax(pred)
                second_prob = pred[j]

                # third input, create a vector of the first and second indices, ex. (A, A, [end])
                # to get the probability for index k
                one_hot = torch.zeros([1, 100, 20]).to(device)
                one_hot[0][98][j] += 1
                one_hot[0][97][i] += 1

                pred = model(one_hot)
                pred = pred[-1]
                pred = nn.functional.softmax(pred)
                third_prob = pred[k]

                model_probs[i][j][k] = first_prob * second_prob * third_prob

model_probs = torch.flatten(model_probs)

plt.plot(model_probs)
plt.show()

sequence_count = torch.Tensor(sequence_count)
distance = torch.linalg.norm(sequence_count - model_probs)

print("Distance: {}".format(distance))

difference = torch.abs(sequence_count - model_probs)
difference = torch.sort(difference)

furthest_elements = difference[0][-20:].numpy()
furthest_elements_index = difference[1][-20:].numpy()

print('furthest probs: {}'.format(furthest_elements))

closest_elements = difference[0][:20].numpy()
closest_elements_index = difference[1][:20].numpy()

print("closest probs: {}".format(closest_elements))

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
encodings = np.eye(len(amino_acids))

amino_acid_encodings = {}
for i, amino_acid in enumerate(amino_acids):
    amino_acid_encodings[i] = amino_acid

furthest_sequences = []
for index in furthest_elements_index:
    amino_acid = ''
    cur_index = np.unravel_index(index, (20, 20, 20))
    amino_acid += amino_acid_encodings[cur_index[0]]
    amino_acid += amino_acid_encodings[cur_index[1]]
    amino_acid += amino_acid_encodings[cur_index[2]]
    furthest_sequences.append(amino_acid)

closest_sequences = []
for index in closest_elements_index:
    amino_acid = ''
    cur_index = np.unravel_index(index, (20, 20, 20))
    amino_acid += amino_acid_encodings[cur_index[0]]
    amino_acid += amino_acid_encodings[cur_index[1]]
    amino_acid += amino_acid_encodings[cur_index[2]]
    closest_sequences.append(amino_acid)

print('furthest: {}'.format(furthest_sequences))
print('closest: {}'.format(closest_sequences))