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

with torch.no_grad():
    d_lens = []
    for sequence in tqdm(val_set):
        target = np.argmax(sequence[-2])  # ignore EOS
        sequence = np.array(sequence[:-2])

        # Get initial prediction probability
        sequence_tensor = np.expand_dims(sequence, axis=0)
        sequence_tensor = torch.Tensor(sequence_tensor).to(device)
        pred = model(sequence_tensor)
        pred = pred.cpu().numpy()[-1]
        pred = np.exp(pred) / np.sum(np.exp(pred))
        target_prob = np.max(pred)

        for k in range(1, len(sequence) + 1):
            changed_value = np.copy(sequence[-k])
            idx = np.argmax(changed_value)
            while True:
                replacement = np.random.randint(0, 20)
                if replacement != idx:
                    break

            sequence[-k] = one_hots[replacement]
            sequence_tensor = np.expand_dims(sequence, axis=0)
            sequence_tensor = torch.Tensor(sequence_tensor).to(device)
            pred = model(sequence_tensor)
            pred = pred.cpu().numpy()[-1]
            pred = np.exp(pred) / np.sum(np.exp(pred))
            if np.abs(pred[target] - target_prob) < .05:
                d_lens.append(k)
                break

            sequence[-k] = changed_value

    d_lens = np.array(d_lens)
    print('Mean:', np.average(d_lens))
    print('Max:', np.max(d_lens))
