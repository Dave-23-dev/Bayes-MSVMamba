import torch
import torchvision
from torchvision import datasets, transforms
import os, pickle, numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

root='./Experiment/data'


# train_dataset = datasets.MNIST(
#     root=root,      # 数据集将被下载到当前目录下的 'data' 文件夹
#     train=True,         # 指定为训练集
#     download=True,      # 如果本地没有数据，自动下载
# )

# test_dataset = datasets.MNIST(
#     root=root,
#     train=False,        # 指定为测试集
#     download=True,
#     transform=transforms.ToTensor()
# )

trainset = torchvision.datasets.CIFAR10(
    root=root+'/CIFAR10',  # 存储路径
    train=True,
    download=True,  # 首次设为 True
    transform=transforms.ToTensor()
)

testset = torchvision.datasets.CIFAR10(
    root=root+'/CIFAR10',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


trainset100 = torchvision.datasets.CIFAR100(
    root=root+'/CIFAR100',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

trainset100 = torchvision.datasets.CIFAR100(
    root=root+'/CIFAR100',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

dataset = CIFAR10(root='./Experiment/data/CIFAR10', train=True, download=True)
os.makedirs('./Experiment/data/CIFAR10/train', exist_ok=True)

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for label_name in label_names:
    os.makedirs(f'./Experiment/data/CIFAR10/train/{label_name}', exist_ok=True)

print('转换中...')
for i, (img, label) in enumerate(dataset):
    if i % 1000 == 0: print(f'已转换 {i}')
    label_name = label_names[label]
    img.save(f'./Experiment/data/CIFAR10/train/{label_name}/{i}.png')

print('训练集转换完成！')