import pickle
import numpy as np
import torch.utils.data


class CifarTrain(torch.utils.data.Dataset):
    def __init__(self):
        images = []
        labels = []
        for i in range(1, 6):
            with open('../cifar10/data_batch_' + str(i), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                images.append(data[b'data'])
                labels.append(data[b'labels'])

        self.x = np.reshape(np.asarray(images), (-1, 3, 32, 32))
        self.y = np.asarray(labels).flatten()

    def __getitem__(self, item):
        x = torch.from_numpy(self.x[item]).float()
        y = self.y[item]

        return x, y

    def __len__(self):
        return len(self.y)


class CifarTest(torch.utils.data.Dataset):
    def __init__(self):
        images = []
        labels = []
        with open('../cifar10/test_batch', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            images.append(data[b'data'])
            labels.append(data[b'labels'])

        self.x = np.reshape(np.asarray(images), (-1, 3, 32, 32))
        self.y = np.asarray(labels).flatten()

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]

        x = torch.from_numpy(x)

        return x.float(), y

    def __len__(self):
        return len(self.y)
