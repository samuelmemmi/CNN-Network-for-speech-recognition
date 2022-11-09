import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from gcommand_dataset import GCommandLoader

import os
import os.path

import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_average_loss = []
train_average_accuracy = []
validation_average_accuracy = []
validation_average_loss = []
classes = ('bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
           'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
           'up', 'wow', 'yes', 'zero')
# Hyper-parameters
num_epochs = 14
batch_size = 10
learning_rate = 0.01
epochs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

path_train = 'gcommands/train/'
path_valid = 'gcommands/valid/'
path_test = 'gcommands/test'

train_dataset = GCommandLoader(path_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

validation_dataset = GCommandLoader(path_valid)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                                                pin_memory=True)

test_set = GCommandLoader(path_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                          pin_memory=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1920, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.conv6(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train():
    model.train()
    train_loss = 0
    correct = 0
    # iterate once over training_loader (1 epoc)
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        optimizer.zero_grad()
        output = model(train_x)
        output = output.to(device)
        loss = F.nll_loss(output, train_y)
        loss = loss.to(device)
        loss.backward()
        optimizer.step()
        # calculate loss and accuracy for report file
        train_loss += loss
        train_loss = train_loss.to(device)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(train_y.view_as(pred)).cpu().sum()
        correct = correct.to(device)

    train_loss /= len(train_loader.dataset) / batch_size
    train_average_loss.append(loss)
    accuracy = 100. * correct / len(train_loader.dataset)
    train_average_accuracy.append(accuracy)
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(train_loader.dataset),
        accuracy))


def validation_test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (validation_x, validation_y) in enumerate(validation_loader):
            validation_x = validation_x.to(device)
            validation_y = validation_y.to(device)
            output = model(validation_x)
            output = output.to(device)
            test_loss += F.nll_loss(output, validation_y, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(validation_y.view_as(pred)).cpu().sum()

        test_loss /= len(validation_loader.dataset)
        validation_average_loss.append(test_loss)
        accuracy = 100. * correct / len(validation_loader.dataset)
        validation_average_accuracy.append(accuracy)
        print('Validation set : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(
            validation_loader.dataset), accuracy))


def test():
    file_names = test_loader.sampler.data_source.file_names
    file_names.remove('.DS_Store')
    i = 0
    prediction = []
    model.eval()
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            prediction.append(file_names[i] + "," + classes[pred.item()])
            i += 1
        with open("test_y", "w") as f:
            for j in range(len(prediction)):
                f.write(prediction[j])
                # In the last prediction do not add a blank line
                if not j == len(prediction) - 1:
                    f.write('\n')


def showPlotAccuracy():
    # create and show graph average accuracy
    plt.plot(epochs_list, train_average_accuracy, 'g', label='Training accuracy')
    plt.plot(epochs_list, validation_average_accuracy, 'b', label='Validation accuracy')
    plt.title('Average accuracy per epoch for validation and training set')
    plt.xlabel('Epochs')
    plt.ylabel('Average_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs)):
        train()
        validation_test()
    showPlotAccuracy()
    test()
