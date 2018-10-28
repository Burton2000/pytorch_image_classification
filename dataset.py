from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MnistDataset(Dataset):
    def __init__(self, x, y=None):
        self.data = x
        self.labels = y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]


def load_train_csv_dataset(train_csv_path, validation_percent=0):
    train_data_table = pd.read_csv(train_csv_path)

    # Separate the labels from the input data.
    train_y = train_data_table.values[:, 0]
    train_x = train_data_table.values[:, 1:].astype(np.float32)

    # Calculate how much of our training data is for train and validation.
    num_train = len(train_y)
    num_val = int(num_train * validation_percent)

    # Reshape data back to images, transpose to N,C,H,W format for pytorch.
    train_x = train_x.reshape([-1, 28, 28, 1]).transpose((0, 3, 1, 2))

    # Split for train/val.
    val_x = train_x[0:num_val]
    val_y = train_y[0:num_val]
    train_x = train_x[num_val:]
    train_y = train_y[num_val:]

    return train_x, train_y, val_x, val_y


def load_test_csv_dataset(csv_path):
    test_data_table = pd.read_csv(csv_path)
    test_x = test_data_table.values.astype(np.float32)

    test_x = test_x.reshape([-1, 28, 28, 1]).transpose((0, 3, 1, 2))

    return test_x
