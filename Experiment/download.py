import torch
from torchvision import datasets, transforms


train_dataset = datasets.MNIST(
    root='./Experiment/dataset/Mnist',      # 数据集将被下载到当前目录下的 'data' 文件夹
    train=True,         # 指定为训练集
    download=True,      # 如果本地没有数据，自动下载
)

test_dataset = datasets.MNIST(
    root='./Experiment/dataset/Mnist',
    train=False,        # 指定为测试集
    download=True,
    transform=transforms.ToTensor()
)