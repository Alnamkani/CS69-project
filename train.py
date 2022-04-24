import argparse
import logging
import numpy as np
import torch

import cifar10, Model, MiniBatcher, utils



def one_hot(y, n_classes):
    """Encode labels into ont-hot vectors
    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot


def main(*ARGS):    
    #load data
    folder = "./data"
    X_train, y_train = cifar10.load_train_data(folder, max_n_examples=-1)
    X_test, y_test = cifar10.load_test_data(folder)

    # one hot encode data
    y_train_1hot = one_hot(y_train, cifar10.N_CLASSES)
    y_test_1hot = one_hot(y_test, cifar10.N_CLASSES)

    # to torch tensor
    X_train, y_train, y_train_1hot = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(y_train_1hot)
    X_train = X_train.type(torch.FloatTensor)
    X_test, y_test, y_test_1hot = torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(y_test_1hot)
    X_test = X_test.type(torch.FloatTensor)

    model = Model.Net()

    losses = []
    train_accs = []
    test_accs = []
    batch_size = 24
    # wihtout minibatch size, this only shuffles the indices
    batcher = MiniBatcher(batch_size, X_train.shape[0])

    for i_epoch in range(20):

        for train_idxs in batcher.get_one_batch():

            # fit to the training data
            loss = model.train_one_epoch(X_train[train_idxs], y_train[train_idxs], y_train_1hot[train_idxs], 0.01)

            # monitor training and testing accuracy
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = utils.accuracy(y_train, y_train_pred)
            test_acc = utils.accuracy(y_test, y_test_pred)
            logging.info("Accuracy(train) = {}".format(train_acc))
            logging.info("Accuracy(test) = {}".format(test_acc))

        # collect results for plotting for each epoch
        loss = model.loss(X_train, y_train, y_train_1hot)
        losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)



if __name__ == '__main__':
    main()