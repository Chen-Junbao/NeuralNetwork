import pickle
import numpy as np
from PIL import Image
import torch.utils.data

from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class CifarTrain(torch.utils.data.Dataset):
    def __init__(self):
        images = []
        labels = []
        for i in range(1, 6):
            with open('../cifar10/data_batch_' + str(i), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                images += list(data[b'data'])
                labels += list(data[b'labels'])

        self.x = np.reshape(images, (50000, 3, 32, 32))
        self.y = np.asarray(labels).flatten()

    def __getitem__(self, item):
        x = self.x[item]
        x = np.transpose(x, (1, 2, 0))
        img = Image.fromarray(np.uint8(x))
        x = transform_train(img).float()
        y = self.y[item]

        return x, y

    def __len__(self):
        return len(self.y)


class CifarTest(torch.utils.data.Dataset):
    def __init__(self):
        with open('../cifar10/test_batch', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            images = list(data['data'])
            labels = list(data['labels'])

        self.x = np.reshape(images, (10000, 3, 32, 32))
        self.y = np.asarray(labels).flatten()

    def __getitem__(self, item):
        x = self.x[item]
        x = np.transpose(x, (1, 2, 0))
        img = Image.fromarray(np.uint8(x))
        x = transform_test(img).float()
        y = self.y[item]

        return x, y

    def __len__(self):
        return len(self.y)
