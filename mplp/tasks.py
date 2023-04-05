import random
import math
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from network import *
from layers import *


# Base class
class Task:
    
    def __init__(self, batch_size, device=torch.device("cpu")):
        self.batch_size = batch_size
        self.epoch = 0
        self.device = device
        self.index = 0
        return

    def get_next_train_batch(self, update_index=True, n=None):
        # get next batch -> if at the end of the iterator reset the iterator and increase the epoch count
        # returns batch, epoch
        return self._get_next(update_index, n=n)

    
    def _get_next(self, update_index=True, n=None):

        if self.index + self.batch_size >= len(self.X_train):
            self.index = 0
            self.epoch += 1
            # raise StopIteration

        batch = self.X_train[self.index:self.index+self.batch_size], self.y_train[self.index:self.index+self.batch_size], self.epoch

        if update_index:
            self.index += self.batch_size

        return batch




# Toy problems only, func needs to be implemented to inherit from this class
class FuncTask(Task):
    def __init__(self, params, start=-5, stop=5, num_examples=1024, batch_size=32, test_size=0, device=torch.device("cpu")):
        super().__init__(batch_size=batch_size)

        self.start = start
        self.stop = stop
        self.params = params
        self.device = device

        X, y = self.generate_data(start, stop, num_examples, device, **params)

        # X = torch.unsqueeze(X, dim=-1)
        # y = torch.unsqueeze(y, dim=-1)


        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)
        else:
            X_train = X
            y_train = y
            # empty tensor
            X_test = torch.empty(0)
            y_test = torch.empty(0)

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.calc_max_loss()
        self.min_loss = 0.005

    def generate_data(self, start, stop, num_data, device, **params):
        x = torch.rand((num_data, self.input_size), device=device) * (stop-start) + start
        return x, self.func(x, **params)

    def sample_data(self, n_steps, batch_size):
        # returns a loader with the data to train on
        return self.random_factory(n_steps*batch_size, batch_size).train_loader
    
    
    def plot(self, network, n_points=100, **kwargs):
        class_name = self.__class__.__name__
        title = class_name + " " + ", ".join(["{} = {:.2f}".format(key, param) for key, param in self.params.items()])
        
        if self.input_size == 1:
            x = torch.linspace(self.start, self.stop, n_points).unsqueeze(-1).cpu()
            y = self.func(x,**self.params).cpu()

            
            plt.clf()
            plt.plot(x, y, **kwargs)
            
            if network:
                guesses = network.forward(x)
                plt.plot(x, guesses, **kwargs)

            plt.legend()
            plt.title(title)
            return 
        elif self.input_size == 2:
            n_points = int(math.sqrt(n_points))
            # same as above but for 3d
            x = torch.linspace(self.start, self.stop, n_points).cpu()
            x2 = torch.linspace(self.start, self.stop, n_points).cpu()
            meshgrid = torch.meshgrid(x, x2)
            meshgrid = torch.stack(meshgrid, dim=-1)
            meshgrid = meshgrid.view(-1, 2)
            
            y = self.func(meshgrid, **self.params).cpu()
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(meshgrid[:,0], meshgrid[:, 1], y)

            if network:
                guesses = network.forward(meshgrid)
                ax.scatter(meshgrid[:,0], meshgrid[:, 1], guesses)
                
            return
        else:
            raise NotImplementedError

    def calc_max_loss(self):
        # to get the max loss we assume the network is a constant function

        def const_func(x): # this is the most simple model, a straight line
            return torch.zeros_like(x)

        n=10

        x = torch.rand((self.batch_size*n, self.input_size)) * (self.stop-self.start) + self.start
        y = self.func(x, **self.params)
        pred = const_func(x)
        loss = MessagePassingMSE.loss(pred, y)
        self.max_loss = loss.item()
    
    def normalize_loss(self, losses):
        normalized = (losses - self.min_loss) / (self.max_loss - self.min_loss)
        return normalized


class CosLinTask(FuncTask):

    def __init__(self, *args, **kwargs):
        self.input_size = 1
        self.output_size = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def func(x, c, d):
        return torch.cos(c*x) + d*x

    @staticmethod
    def random_factory(num_examples, batch_size, device=torch.device("cpu")):

        crange = 0, 3
        c = random.random() * (crange[1]-crange[0]) + crange[0]

        drange = -2, 2
        d = random.random() * (drange[1]-drange[0]) + drange[0]
        return CosLinTask(start=-5, stop=5, num_examples=num_examples, batch_size=batch_size, device=device, params={"c": c, "d": d})


class SinCosLinTask(FuncTask):

    def __init__(self, *args, **kwargs):
        self.input_size = 1
        self.output_size = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def func(x, a, b, c):
        return torch.sin(a*x) + torch.cos(b*x) + c*x

    @staticmethod
    def random_factory(num_examples, batch_size, device=torch.device("cpu")):

        arange = 0.5, 0.5*3.141592
        a = random.random() * (arange[1]-arange[0]) + arange[0]

        brange = 0.5, 0.5*3.141592
        b = random.random() * (brange[1]-brange[0]) + brange[0]

        crange = -1, 1
        c = random.random() * (crange[1]-crange[0]) + crange[0]

        return SinCosLinTask(start=-5, stop=5, num_examples=num_examples, batch_size=batch_size, device=device, params={"a": a, "b": b, "c": c})


class SinTask(FuncTask):

    def __init__(self, *args, **kwargs):
        self.input_size = 1
        self.output_size = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def func(x, amp, phase):
        return amp * torch.sin(x+phase)

    @staticmethod
    def random_factory_amplitude_phase(num_examples, batch_size, device=torch.device("cpu")):

        amp_range = 0.1, 0.5
        amp = random.random() * (amp_range[1]-amp_range[0]) + amp_range[0]

        phase_range = 0, 3.1415
        phase = random.random() * (phase_range[1]-phase_range[0]) + phase_range[0]

        return SinTask(start=-5, stop=5, num_examples=num_examples, batch_size=batch_size, device=device, params={"amp": amp, "phase": phase})

    @staticmethod
    def random_factory(num_examples, batch_size, device=torch.device("cpu"), random_amplitude=False):
        
        if not random_amplitude:
            amp = 1.0
        else:
            amp_range = 1.0, 5.0
            amp = random.random() * (amp_range[1]-amp_range[0]) + amp_range[0]

        phase_range = 0, 3.1415
        phase = random.random() * (phase_range[1]-phase_range[0]) + phase_range[0]

        return SinTask(start=-5, stop=5, num_examples=num_examples, batch_size=batch_size, device=device, params={"amp": amp, "phase": phase})


class MultiplyTask(FuncTask):

    def __init__(self, *args, **kwargs):
        self.input_size = 2
        self.output_size = 1
        super().__init__(*args, **kwargs)
        

    @staticmethod
    def func(x):
        return torch.unsqueeze(x[:, 0] * x[:, 1], 1)

    @staticmethod
    def random_factory(num_examples, batch_size, device=torch.device("cpu")):
        return MultiplyTask(start=-5, stop=5, num_examples=num_examples, batch_size=batch_size, device=device, params={})


class LogCosTask(FuncTask):

    def __init__(self, *args, **kwargs):
        self.input_size = 1
        self.output_size = 1
        super().__init__(*args, **kwargs)


    @staticmethod
    def func(x, a, flip):
        res = (torch.log10(x**2 + 1.0 + torch.cos(a*x)) - 0.5)
        if flip:
            return -res
        else:
            return res

    @staticmethod
    def random_factory(num_examples, batch_size, device=torch.device("cpu")):
        a = random.random() * 4.5
        flip = random.random() > 0.5
        return LogCosTask(start=-5, stop=5, num_examples=num_examples, batch_size=batch_size, device=device, params={'a': a, 'flip': flip})


class LogSinTask(FuncTask):

    def __init__(self, *args, **kwargs):
        self.input_size = 1
        self.output_size = 1
        super().__init__(*args, **kwargs)


    @staticmethod
    def func(x, a, flip):
        res = (torch.log10(x**2 + 1.0 + torch.sin(a*x)) -0.5)/1.5
        if flip:
            return -res
        else:
            return res

    @staticmethod
    def random_factory(num_examples, batch_size, device=torch.device("cpu")):
        a = random.random() * 9 - 4.5
        flip = random.random() > 0.5
        return LogSinTask(start=-5, stop=5, num_examples=num_examples, batch_size=batch_size, device=device, params={'a': a, 'flip': flip})






class MNISTTask(Task):

    def __init__(self, batch_size=128, device=torch.device("cpu")):
        super().__init__(batch_size=batch_size, device=device)

        self.input_size = 28*28
        self.output_size = 10
        self.number_of_inner_steps = 200
        self.mean = 0
        self.std = 1

        # make caching system because otherwise it is slow
        check_if_exists = ['data/MNIST/X_train.pt', 'data/MNIST/y_train.pt', 'data/MNIST/X_test.pt', 'data/MNIST/y_test.pt']
        use_cache = True
        for file in check_if_exists:
            if not os.path.exists(file):
                use_cache = False
                break
        
        if use_cache:
            self.X_train = torch.load('data/MNIST/X_train.pt')
            self.y_train = torch.load('data/MNIST/y_train.pt')
            self.X_test = torch.load('data/MNIST/X_test.pt')
            self.y_test = torch.load('data/MNIST/y_test.pt')
        else:
            transform = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                            torchvision.transforms.Lambda(lambda x: torch.flatten(x).to(device))
                                            ])

            target_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Lambda(lambda y: torch.tensor(y).to(device))
                                    ])
            
            train_dataset = torchvision.datasets.MNIST('data/', train=True, download=True, transform=transform, target_transform=target_transform)

            self.train_loader = torch.utils.data.DataLoader(
                                            train_dataset,
                                            batch_size=2048, shuffle=False
                                            )

            self.train_iter = iter(self.train_loader)

            test_dataset = torchvision.datasets.MNIST('data/', train=False, download=True, transform=transform, target_transform=target_transform)

            self.test_loader = torch.utils.data.DataLoader(
                                            test_dataset,
                                            batch_size=2048, shuffle=False
                                            )

            self.test_iter = iter(self.test_loader)

            # convert the train_loader to a single tensor
            self.X_train = torch.cat([x[0] for x in self.train_loader])
            self.y_train = torch.cat([x[1] for x in self.train_loader])

            self.X_test = torch.cat([x[0] for x in self.test_loader])
            self.y_test = torch.cat([x[1] for x in self.test_loader])

            # save to pt file
            torch.save(self.X_train, 'data/MNIST/X_train.pt')
            torch.save(self.y_train, 'data/MNIST/y_train.pt')
            torch.save(self.X_test, 'data/MNIST/X_test.pt')
            torch.save(self.y_test, 'data/MNIST/y_test.pt')


         # create permuation for train and the test set
        self.permutation_train = torch.randperm(self.X_train.shape[0])
        self.permutation_test = torch.randperm(self.X_test.shape[0])

        # apply the permuations
        self.X_train = self.X_train[self.permutation_train]
        self.y_train = self.y_train[self.permutation_train]

        self.X_test = self.X_test[self.permutation_test]
        self.y_test = self.y_test[self.permutation_test]
    
    def set_random_norm(self):
        # mean random float between -0.5 and 0.5
        self.mean = random.random() - 0.5
        # std random float between 0.5 and 1
        self.std = random.random() * 0.5 + 0.5

    def _get_next(self, update_index=True, n=None):
        if n is None:
            n = 1

        if self.index + self.batch_size*n >= len(self.X_train):
            self.index = 0
            self.epoch += 1

        batch = (self.X_train[self.index:self.index+self.batch_size*n]-self.mean)/self.std, self.y_train[self.index:self.index+self.batch_size*n], self.epoch

        if update_index:
            self.index += self.batch_size*n

        return batch


class FashionMNISTTask(Task):

    def __init__(self, batch_size=128, device=torch.device("cpu")):
        super().__init__(batch_size=batch_size, device=device)

        self.input_size = 28*28
        self.output_size = 10
        self.number_of_inner_steps = 200
        self.mean = 0
        self.std = 1

        check_if_exists = ['data/FashionMNIST/X_train.pt', 'data/FashionMNIST/y_train.pt', 'data/FashionMNIST/X_test.pt', 'data/FashionMNIST/y_test.pt']
        use_cache = True
        for file in check_if_exists:
            if not os.path.exists(file):
                use_cache = False
                break
        
        if use_cache:
            self.X_train = torch.load('data/FashionMNIST/X_train.pt')
            self.y_train = torch.load('data/FashionMNIST/y_train.pt')
            self.X_test = torch.load('data/FashionMNIST/X_test.pt')
            self.y_test = torch.load('data/FashionMNIST/y_test.pt')
        else:
            transform = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.2860,), (0.3530,)),
                                            torchvision.transforms.Lambda(lambda x: torch.flatten(x).to(device))
                                            ])

            target_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Lambda(lambda y: torch.tensor(y).to(device))
                                    ])
            
            train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=transform, target_transform=target_transform)

            self.train_loader = torch.utils.data.DataLoader(
                                            train_dataset,
                                            batch_size=2048, shuffle=False
                                            )

            self.train_iter = iter(self.train_loader)

            test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=transform, target_transform=target_transform)

            self.test_loader = torch.utils.data.DataLoader(
                                            test_dataset,
                                            batch_size=2048, shuffle=False
                                            )

            self.test_iter = iter(self.test_loader)

            # convert the train_loader to a single tensor
            self.X_train = torch.cat([x[0] for x in self.train_loader])
            self.y_train = torch.cat([x[1] for x in self.train_loader])

            self.X_test = torch.cat([x[0] for x in self.test_loader])
            self.y_test = torch.cat([x[1] for x in self.test_loader])

            # save to pt file
            torch.save(self.X_train, 'data/FashionMNIST/X_train.pt')
            torch.save(self.y_train, 'data/FashionMNIST/y_train.pt')
            torch.save(self.X_test, 'data/FashionMNIST/X_test.pt')
            torch.save(self.y_test, 'data/FashionMNIST/y_test.pt')



    
        # create permuation for train and the test set
        self.permutation_train = torch.randperm(self.X_train.shape[0])
        self.permutation_test = torch.randperm(self.X_test.shape[0])

        # apply the permuations
        self.X_train = self.X_train[self.permutation_train]
        self.y_train = self.y_train[self.permutation_train]

        self.X_test = self.X_test[self.permutation_test]
        self.y_test = self.y_test[self.permutation_test]

    def _get_next(self, update_index=True, n=None):
        if n is None:
            n = 1

        if self.index + self.batch_size*n >= len(self.X_train):
            self.index = 0
            self.epoch += 1

        batch = self.X_train[self.index:self.index+self.batch_size*n], self.y_train[self.index:self.index+self.batch_size*n], self.epoch

        if update_index:
            self.index += self.batch_size*n

        return batch



if __name__ == "__main__":
    import matplotlib.pyplot as plt


    device = torch.device('cpu')

    task = MNISTTask(device=device)
