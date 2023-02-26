
# '''
# This is the example code of benign training and poisoned training on torchvision.datasets.DatasetFolder.
# Dataset is CIFAR-10.
# Attack method is BadNets.
# '''
import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip

import core


from torch.utils.data import DataLoader

dataset = torchvision.datasets.DatasetFolder



if __name__ == '__main__':
    # download dataset
    # train_data = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=True, download=True)
    # tes_data = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, download=True)
    # prepossessing dataset
    # 将原始的文件变成png格式的文件


    # image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> torch.Tensor -> network input
    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip()
    ])
    transform_test = Compose([
        ToTensor()
    ])

    trainset = dataset(
        root='./data/cifar10/train/',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)


    testset = dataset(
        root='./data/cifar10/test',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    # the parameters of model
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    badnets = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(18),
        # model=core.models.BaselineMNISTNetwork(),
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        poisoned_transform_train_index=0,
        poisoned_target_transform_index=0,
        schedule=None,
        seed=666
    )

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'benign_training': False,
        'batch_size': 128,
        'num_workers': 0,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 5,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'train_benign_DatasetFolder-CIFAR10'
    }

    badnets.train(schedule)
    # Test attacked Model
    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '1',
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,

        'save_dir': 'experiments',
        # 'experiment_name': 'test_benign_CIFAR10_Blended'
        'experiment_name': 'test_poisoned_CIFAR10_BadNets'
    }
    _, poisoned_testset = badnets.get_poisoned_dataset()
    badnets.test(schedule,test_dataset=testset, poisoned_test_dataset= poisoned_testset)
