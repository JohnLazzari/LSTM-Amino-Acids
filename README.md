# Project_2
Programming assignment 2 based on RNNs.

## Creating the datasets
We used `sequences.py` to create the datasets. The script has two optional flags: `--output-dir DIR` specifies where to save the datasets and `--full` processes the entire dataset (with the default being 1%). The script creates the files `train.npy` and `val.npy`.

## Training the model

## Experiments
The experiments were run using the following scripts.

### Long Term Dependencies
Long term dependencies were tested using `long-term-test.py`. The script loads the trained model. Then, it iteratively changes an amino acid in the sequence `k` positions from the prediction until a threshold of `10%` change in the prediction probability is not broken.

### Correctly Predicted Sequences
The table in Problem 3.2 was generated using `first-k-test.py`. The script iterates through the validation dataset to predict the rest of a sequence given the first`k` amino acids. The amount of correct predictions are recorded in the table. 
