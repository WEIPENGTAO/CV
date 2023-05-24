import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


class Config:
    batch_size = 64
    epoch = 10
    momentum = 0.9
    alpha = 1e-3

    print_per_step = 100


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TrainProcess:
    def __init__(self):
        self.train, self.test = self.load_data()
        self.net = LeNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=Config.alpha, momentum=Config.momentum)

    @staticmethod
    def load_data():
        print("Loading Data...")
        train_data = datasets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
        test_data = datasets.MNIST(root='./data/',
                                   train=False,
                                   transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=Config.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=Config.batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    def train_step(self):
        steps = 0
        start_time = datetime.now()
        print("Training & Evaluating...")
        for epoch in range(Config.epoch):
            print("Epoch {:3}".format(epoch + 1))
            for data, label in self.train:
                self.optimizer.zero_grad()
                outputs = self.net(data)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                if steps % Config.print_per_step == 0:
                    _, predicted = torch.max(outputs, 1)
                    correct = int(sum(predicted == label))
                    accuracy = correct / Config.batch_size
                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    print(msg.format(steps, loss, accuracy, time_usage))
                steps += 1

        test_loss = 0.
        test_correct = 0
        for data, label in self.test:
            outputs = self.net(data)
            loss = self.criterion(outputs, label)
            test_loss += loss * Config.batch_size
            _, predicted = torch.max(outputs, 1)
            correct = int(sum(predicted == label))
            test_correct += correct
        accuracy = test_correct / len(self.test.dataset)
        loss = test_loss / len(self.test.dataset)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))
        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))

        # 保存模型参数
        torch.save(self.net.state_dict(), 'model.pth')


if __name__ == "__main__":
    p = TrainProcess()
    p.train_step()
