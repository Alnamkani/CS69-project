import argparse
import logging
from   pyexpat import model
from   statistics import mode
import numpy as np
import torch
from   torchvision import datasets, transforms
import cifar10, Model, MiniBatcher, utils
import matplotlib.pyplot as plt
import torchvision


def one_hot(y, n_classes):
    """Encode labels into ont-hot vectors
    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def main(*ARGS):    
    #load data
    folder = "./data"

    batch_size = 64

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=folder, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=folder, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
    num_e = 10

    model = Model.Net()

    for epochs in range(num_e):

        for i, data in enumerate(trainloader, 0):

            input, label = data

            model.train_one_epoch(input, label, None, None)

    print("done training")

    path = "MA_weights_64.ptnnp"

    torch.save(model.model.state_dict(), path)

    # for i, data in enumerate(trainloader, 0):
    #     input, label = data
    #     lo = model.loss(input, label, None)
    #     lo.backward()
    #     for x in model.model.parameters():
    #         print(x.grad.data.size())
    #     print(lo)
    #     break
    
    # return 

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            # outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            predicted = model.predict(images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':
    main()