import torchvision.datasets as datasets

# 加载MNIST测试数据集
mnist_test = datasets.MNIST(root='./data/',
                            train=False,
                            transform=None,
                            download=True)

# 获取第一张图像和标签
image, label = mnist_test[0]

# 保存图像
image.save("mnist_image.png")
