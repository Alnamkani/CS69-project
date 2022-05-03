import torch
import math

class HMC():
    def __init__(self, model):
        self.model = model
        self.shapes = [param.shape for param in self.model.parameters()]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)

    def sample(self, n, initials, rng, deltas, num_leap):
        samples = []
        pot = []

        for (param, initial) in zip(self.model.parameters(), initials):
            param.data.copy_(initial)
        with torch.no_grad():
            nlf = self.model.loss().item()

        samples.append([param.data for param in self.model.parameters()])
        pot.append(nlf)

        accept_counter = 0
        while accept_counter < n:
            v = [torch.randn(*shape, generator=rng) for (shape, param) in zip(self.shapes, self.model.parameters())]
            nlf0 = pot[-1]
            k0 = sum(0.5 * torch.sum(vel ** 2).item() for vel in v)

            for j in range(num_leap):
                self.optimizer.zero_grad()
                self.model.loss().backward()
                for pram, delta, vel in zip(self.model.parameters(), deltas, v):
                    vel -= torch.mul(delta / 2, pram.grad.data)
                for pram, delta, vel in zip(self.model.parameters(), deltas, v):
                    pram.data += torch.mul(delta, vel)
                self.optimizer.zero_grad()
                self.model.loss().backward()
                for pram, delta, vel in zip(self.model.parameters(), deltas, v):
                    vel -= torch.mul(delta / 2, pram.grad.data)


            with torch.no_grad():
                nlf1 = self.model.loss().item()
            k1 = sum(0.5 * torch.sum(vel ** 2).item() for vel in v)

            a = min(1, math.exp(nlf0 + k0 - nlf1 - k1))
            acc = False
            if torch.rand(1, generator=rng) <= a:
                acc = True

            if acc:
                samples.append([param.data for param in self.model.parameters()])
                pot.append(nlf1)
            else:
                for (param, initial) in zip(self.model.parameters(), samples[-1]):
                    param.data.copy_(initial)
                pot.append(nlf0)
            accept_counter += int(acc)
        return samples