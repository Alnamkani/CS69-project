import logging
import torch
from collections import OrderedDict


class Net:
    def __init__(self, gpu_id=-1):
        net_dict = OrderedDict([("c_1", torch.nn.Conv2d(3, 6, 5))])
        net_dict['r_1'] = torch.nn.ReLU()
        net_dict["p_1"] = torch.nn.MaxPool2d(2, 2)


        net_dict['c_2'] = torch.nn.Conv2d(6, 16, 5)
        net_dict['r_2'] = torch.nn.ReLU()
        net_dict["p_2"] = torch.nn.MaxPool2d(2, 2)

        net_dict['f'] = torch.nn.Flatten()

        net_dict['l_1'] = torch.nn.Linear(16*5*5, 120)
        net_dict['r_3'] = torch.nn.ReLU()

        net_dict['l_2'] = torch.nn.Linear(120, 84)
        net_dict['r_4'] = torch.nn.ReLU()

        net_dict['l_3'] = torch.nn.Linear(84, 10)

        if gpu_id == -1:
            self.model = torch.nn.Sequential(net_dict)
        else:
            self.model = torch.nn.Sequential(net_dict).cuda(gpu_id)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples)
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss
        """
        X_train = X
        y_train = y

        # forward
        y_pred = self.model(X_train)
        loss = self.loss_fn(y_pred, y_train.squeeze())

        # backward
        # self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y.squeeze())
        return loss

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        y_pred = self.model(X)
        return torch.max(y_pred, 1)[1]