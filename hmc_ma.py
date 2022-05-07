import torch
import math
from torchvision import datasets, transforms

import Model


class HMC():
    def __init__(self, model):
        self.model = model
        self.shapes = [param.shape for param in self.model.model.parameters()]
        self.optimizer = torch.optim.SGD(self.model.model.parameters(), lr=0.0)

    def sample(self, n, int_model, rng, deltas, num_leap, trainloader):
        samples = []
        pot = []

        # for (param, initial) in zip(self.model.model.parameters(), int_model.model.parameters()):
        #     param.data.copy_(initial)

        self.model.model.load_state_dict(int_model.model.state_dict())

        nlf = 0
        self.optimizer.zero_grad()
        with torch.no_grad():
                for i, data in enumerate(trainloader, 0):
                    input, label = data
                    nlf += self.model.loss(input, label, None).item()

        samples.append(self.model)
        pot.append(nlf)

        # return [samples[0], samples[0]]
        # tmp = Model.Net()
        # tmp.model.load_state_dict(int_model.model.state_dict())

        proposed_model = Model.Net()
        proposed_model.model.load_state_dict(int_model.model.state_dict())

        accept_counter = 0
        while accept_counter < n:
            v = [torch.randn(*shape, generator=rng) for (shape, param) in zip(self.shapes, self.model.model.parameters())]
            nlf0 = pot[-1]
            k0 = sum(0.5 * torch.sum(vel ** 2).item() for vel in v)

            for j in range(num_leap):

                self.optimizer.zero_grad()
                for i, data in enumerate(trainloader, 0):
                    input, label = data
                    self.model.loss(input, label, None).backward()

                for pram, delta, vel in zip(self.model.model.parameters(), deltas, v):
                    vel.data -= torch.mul(delta / 2, pram.grad.data)

                for pram, delta, vel in zip(self.model.model.parameters(), deltas, v):
                    pram.data += torch.mul(delta, vel)

                self.optimizer.zero_grad()
                for i, data in enumerate(trainloader, 0):
                    input, label = data
                    self.model.loss(input, label, None).backward()

                for pram, delta, vel in zip(self.model.model.parameters(), deltas, v):
                    vel.data -= torch.mul(delta / 2, pram.grad.data)

            nlf1 = 0

            self.optimizer.zero_grad()
            with torch.no_grad():
                for i, data in enumerate(trainloader, 0):
                    input, label = data
                    nlf1 += self.model.loss(input, label, None).item()

                # nlf1 = self.model.loss(input, output, None).item()
            k1 = sum(0.5 * torch.sum(vel ** 2).item() for vel in v)

            a = min(1, math.exp(nlf0 + k0 - nlf1 - k1))
            acc = False
            if torch.rand(1, generator=rng) <= a:
                acc = True

            if acc:

                tmp = Model.Net()
                tmp.model.load_state_dict(self.model.model.state_dict())
                samples.append(tmp)
                # samples.append([param.data for param in self.model.model.parameters()])
                pot.append(nlf1)
            else:
                # for (param, initial) in zip(self.model.model.parameters(), samples[-1]):
                #     param.data.copy_(initial)
                print("GG")
                self.model.model.load_state_dict(int_model.model.state_dict())
                pot.append(nlf0)
            accept_counter += int(acc)
        return [proposed_model] + samples # + [tmp] 

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
    model.model.load_state_dict(torch.load("MA_weights.ptnnp"))


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
    samples = h.sample(1, model, thrng, [0.001, 0.01, 0.1], 2, trainloader)

    print(len(samples))


    # # You can immediately evaluate the sampled parameters using
    for x in samples:
        # for (param, sample) in zip(h.model.model.parameters(), x):
        #     param.data.copy_(sample)

        model.model.load_state_dict(x.model.state_dict())

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
        # break
    # Then evaluate model that has parameters sample[i]