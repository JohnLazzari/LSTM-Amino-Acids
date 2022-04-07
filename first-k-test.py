import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from LSTM_train import LSTM
from tqdm import tqdm

device = 'cuda:0'
val_set = np.load('val.npy', allow_pickle=True)
end_token = torch.zeros([1, 20])
one_hots = np.eye(20)

model = LSTM(20, 60, 4).to(device)
model.load_state_dict(torch.load('lstm.pth'))

table = np.zeros((10, 20))

with torch.no_grad():
    for k in range(1, 11):
        for sequence in tqdm(val_set):
            sequence = np.array(sequence)
            seed = np.array(sequence[:k])
            seed = np.expand_dims(seed, axis=0)  # batch of 1
            seed = torch.Tensor(seed).to(device)

            model.pred_len = len(sequence) - k - 1  # ignore EOS character
            next_acids_tensor = model(seed)
            next_acids = []
            for i in range(len(next_acids_tensor)):
                idx = np.argmax(next_acids_tensor[i].cpu().numpy()[-1])
                next_acids.append(one_hots[idx])

            correct_predictions = 0
            for i, (pred, target) in enumerate(zip(next_acids, sequence[k:])):
                if correct_predictions == 19 or (pred[i] != target[i]).all():
                    table[k-1, correct_predictions] += 1
                    break

                correct_predictions += 1

print(table)
