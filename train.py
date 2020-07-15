from generate_dataset import *
from network.resnext import resnext50
# from apex import amp
from PIL import Image

import torch.nn as nn
import torch.utils.data
import numpy as np


def resnet_preprocess(input_data):
    new_data = []
    for i in range(len(input_data)):
        a = input_data[i].numpy()
        a = np.uint8(np.transpose(a, (1, 2, 0)))
        img = Image.fromarray(a)
        img = img.resize((224, 224))
        a = np.array(img)
        new_data.append(np.transpose(a, (2, 0, 1)))
    new_data = torch.from_numpy(np.asarray(new_data))

    return new_data.float()


def adjust_learning_rate(optimizer, init_lr, epoch):
    learning_rate = init_lr * (0.2 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


if __name__ == "__main__":
    dataset = Cifar100Train()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=28, shuffle=True)

    dataset = Cifar100Test()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss().cuda()

    model = resnext50(num_class=100).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    best_accuracy = 0.0

    epoch = 500
    for i in range(epoch):
        print("epoch:", i)

        model.train()

        for j, data in enumerate(train_loader):
            # adjust_learning_rate(optimizer, 0.1, i)
            x, y = data
            # if the model` is ResNet or MobileNetV2, preprocess the dataset first
            x = resnet_preprocess(x)
            x = x.cuda()
            y = y.cuda()

            x_var = torch.autograd.Variable(x)

            prediction = model(x_var)

            loss = criterion(prediction, y.long())

            optimizer.zero_grad()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            #     print(scaled_loss)
            loss.backward()
            print(loss)
            optimizer.step()

        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()

        model.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                x, y = data
                # if the model is ResNet, preprocess the dataset first
                x = resnet_preprocess(x)
                x = x.cuda()
                y = y.cuda()

                x_var = torch.autograd.Variable(x)

                output = model(x_var)

                prediction = torch.argmax(output, 1)

                correct += (prediction == y.long()).sum().float()
                total += len(y)

            accuracy = (correct / total).cpu().item()
            print("accuracy:", accuracy)
            if accuracy > best_accuracy:
                # save best model
                torch.save(model, "./model/resnext.pth")
                best_accuracy = accuracy
