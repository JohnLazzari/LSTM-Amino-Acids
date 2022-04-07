import numpy as np
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(
    'Generate a train and validation dataset from the pdb_seqres.txt file.')
parser.add_argument('--output-dir', type=str, default='.',
    help='The directory in which to save the datasets.')
parser.add_argument('--full', action='store_true',
    help='Process and save the entire dataset.')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    print('Invalid output directory.')
    exit()

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
encodings = np.eye(len(amino_acids))

amino_acid_encodings = {}
for amino_acid, encoding in zip(amino_acids, encodings):
    amino_acid_encodings[amino_acid] = encoding
amino_acid_encodings['\n'] = np.zeros(shape=(len(amino_acids)))

sequences = []
with open('pdb_seqres.txt', 'r') as pdb_seqres:
    for sequence in tqdm(pdb_seqres.readlines()):
        sequence = list(sequence)

        # One-hot encode sequences.
        for i in range(len(sequence)):
            sequence[i] = amino_acid_encodings[sequence[i]]

        sequences.append(sequence)

print('Splitting dataset...')
train_sequences = []
val_sequences = []
for i, sequence in enumerate(sequences):
    if i % 5 == 4:
        val_sequences.append(sequence)
    else:
        train_sequences.append(sequence)

train_sequences = np.array(train_sequences)
val_sequences = np.array(val_sequences)

if not args.full:
    print('Selecting 1% of dataset...')
    train_sequences = train_sequences[::100]
    val_sequences = val_sequences[::100]

print('Saving dataset...')
train_path = os.path.join(args.output_dir, 'train')
val_path = os.path.join(args.output_dir, 'val')
np.save(train_path, train_sequences)
np.save(val_path, val_sequences)

print(f'Saved to {train_path}.npy and {val_path}.npy.')
