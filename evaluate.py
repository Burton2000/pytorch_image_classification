import torch
import numpy as np

from model import MnistCNN
from dataset import load_test_csv_dataset


def eval_model(model, dataloader):
    model.eval()
    num_correct = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            output = model(x_batch)
            _, output = torch.max(output, dim=1)  # 2nd return provides position of max.

            num_correct += torch.sum(output == y_batch)

        model_accuracy = float(num_correct) / len(dataloader.dataset)  # Conver to float when dividing.
    return model_accuracy


def kaggle_test_evaluation(trained_model_path='./trained_model.pt'):
    test_x = load_test_csv_dataset('./mnist_data/test.csv')
    test_x = torch.from_numpy(test_x)

    # Init model, restore its weights and set to eval mode.
    model = MnistCNN()
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    test_labels = []
    for i in range(len(test_x)):
        test_batch = test_x[i].unsqueeze_(0)

        output = model(test_batch)
        _, output = torch.max(output, dim=1)

        test_labels.append(int(output.data.numpy()))

    print('Kaggle test set results saved to kaggle_test_results.csv')
    np.savetxt("kaggle_test_results.csv", test_labels, delimiter=",")


def main():
    kaggle_test_evaluation()


if __name__ == '__main__':
    main()

