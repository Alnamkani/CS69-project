import torch
import math
from torchvision import datasets, transforms

import Model


class HMC():
    def __init__(self, model):
        self.model = model
        self.shapes = [param.shape for param in self.model.model.parameters()]
        self.optimizer = torch.optim.SGD(self.model.model.parameters(), lr=0.0)

    def sample(self, n, initials, rng, deltas, num_leap, input, output):
        samples = []
        pot = []

        for (param, initial) in zip(self.model.model.parameters(), initials):
            param.data.copy_(initial)
        with torch.no_grad():
            nlf = self.model.loss(input, output, None).item()

        samples.append([param.data for param in self.model.model.parameters()])
        pot.append(nlf)

        # # return [samples[0], samples[0]]
        # tmp = samples[0]

        accept_counter = 0
        while accept_counter < n:
            v = [torch.randn(*shape, generator=rng) for (shape, param) in zip(self.shapes, self.model.model.parameters())]
            nlf0 = pot[-1]
            k0 = sum(0.5 * torch.sum(vel ** 2).item() for vel in v)

            for j in range(num_leap):
                self.optimizer.zero_grad()
                self.model.loss(input, output, None).backward()
                for pram, delta, vel in zip(self.model.model.parameters(), deltas, v):
                    vel -= torch.mul(delta / 2, pram.grad.data)
                for pram, delta, vel in zip(self.model.model.parameters(), deltas, v):
                    pram.data += torch.mul(delta, vel)
                self.optimizer.zero_grad()
                self.model.loss(input, output, None).backward()
                for pram, delta, vel in zip(self.model.model.parameters(), deltas, v):
                    vel -= torch.mul(delta / 2, pram.grad.data)


            with torch.no_grad():
                nlf1 = self.model.loss(input, output, None).item()
            k1 = sum(0.5 * torch.sum(vel ** 2).item() for vel in v)

            a = min(1, math.exp(nlf0 + k0 - nlf1 - k1))
            acc = False
            if torch.rand(1, generator=rng) <= a:
                acc = True

            if acc:
                samples.append([param.data for param in self.model.model.parameters()])
                pot.append(nlf1)
            else:
                for (param, initial) in zip(self.model.model.parameters(), samples[-1]):
                    param.data.copy_(initial)
                pot.append(nlf0)
            accept_counter += int(acc)
        # print(accept_counter, "COUNTER")
        return samples

if __name__ == '__main__':
    folder = "./data"
    batch_size = 64
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=folder, train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    for i, data in enumerate(trainloader, 0):
        input, label = data

    model = Model.Net()
    model.model.load_state_dict(torch.load("MA_weights_64.ptnnp"))


    testset = datasets.CIFAR10(root=folder, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         # calculate outputs by running images through the network
    #         # outputs = model(images)
    #         # the class with the highest energy is what we choose as prediction
    #         predicted = model.predict(images)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    h = HMC(model)
    thrng = torch.Generator("cpu")
    thrng.manual_seed(23)


    # The first arg of the function sample is the number of samples
    samples = h.sample(3, [param.data for param in model.model.parameters()], thrng, [0.001, 0.01, 0.1], 50, input, label)

    # print((samples[0]))

    # print(torch.eq(samples[0][0], samples[1][0]))


    # # You can immediately evaluate the sampled parameters using
    # for x in samples:
    #     print("-----------")
    #     print(x)
        # for (param, sample) in zip(h.model.model.parameters(), x):
        #     param.data.copy_(sample)

        # correct = 0
        # total = 0
        #     # since we're not training, we don't need to calculate the gradients for our outputs
        # with torch.no_grad():
        #     for data in testloader:
        #         images, labels = data
        #             # calculate outputs by running images through the network
        #             # outputs = model(images)
        #             # the class with the highest energy is what we choose as prediction
        #         predicted = model.predict(images)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        # # break
    # Then evaluate model that has parameters sample[i]