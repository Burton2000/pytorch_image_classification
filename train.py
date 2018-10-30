import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from model import MnistCNN
from dataset import load_train_csv_dataset
from dataset import MnistDataset
from evaluate import eval_model

LEARNING_RATE = 0.001
NUM_EPOCHS = 5


def train_model(model, dataloader, loss_function, optimizer, epochs):
    # Switch model to train mode (for things like Batch norm and dropout).
    model.train()

    loss_history = []
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            
            # Zero the gradients.
            optimizer.zero_grad()
            
            # Compute output and loss.
            output = model(x_batch)
            loss = loss_function(output, y_batch)

            loss.backward()
            optimizer.step()

        print('Epoch {} training loss: {}'.format(epoch, loss.item()))  # item() method returns standard python number if tensor has 1 element.
        loss_history.append(loss)

    return loss_history


def main():
    # Load the train data.
    train_csv = './mnist_data/train.csv'
    train_x, train_y, val_x, val_y = load_train_csv_dataset(train_csv, validation_percent=0.1)

    # Create pytorch dataloaders for train and validation sets.
    train_dataset = MnistDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=2)
    val_dataset = MnistDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=200, shuffle=False, num_workers=2)

    # Define model, optimizer and loss function.
    model = MnistCNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    # Train our model.
    train_model(model, train_dataloader, loss_func, optimizer, epochs=NUM_EPOCHS)

    val_accuracy = eval_model(model, val_dataloader)
    print('Validation set accuracy: {}'.format(val_accuracy))

    # Save model weights for inference.
    torch.save(model.state_dict(), 'trained_model.pt')


if __name__ == '__main__':
    main()
