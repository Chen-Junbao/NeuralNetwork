from generate_dataset import *
from network.googlenet import GoogLeNet

import torch.nn as nn
import torch.utils.data

if __name__ == "__main__":
    dataset = CifarTrain()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    dataset = CifarTest()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    criterion = nn.CrossEntropyLoss().cuda()

    model = GoogLeNet().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_accuracy = 0.0

    epoch = 100
    for i in range(epoch):
        print("epoch:", i)

        model.train()

        for j, data in enumerate(train_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()

            x_var = torch.autograd.Variable(x)

            prediction = model(x_var)

            loss = criterion(prediction, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()

        model.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                x, y = data
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
                torch.save(model, "./model/googlenet.pth")
                best_accuracy = accuracy
