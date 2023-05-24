import torch
from torchvision import transforms
from PIL import Image


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(64 * 5 * 5, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def load_model():
    model = LeNet()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # 设置为评估模式
    return model


def preprocess(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    return image


def predict(image_path, model):
    image = preprocess(image_path)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


if __name__ == "__main__":
    model = load_model()
    image_path = 'mnist_image.png'  # 替换为要识别的图像路径
    prediction = predict(image_path, model)
    print("Predicted Digit:", prediction)
