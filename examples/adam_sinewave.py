"""
This examples is just to compare the performance of mplp with Adam.
We just train a netowrk using the adam optimizer.
"""

import torch

import os
import random
from mplp.network import MessagePassingNetwork, unroll_fn
from mplp.models import SGDUpdateFunction, SGDMessageReLUActivation, SGDMessageLinear, LearningRate, SGDMessageMSELoss
from mplp.layers import MessagePassingLinear, MessagePassingReLU, MessagePassingMSE
from mplp.tasks import SinTask
from mplp.util import list_of_dicts_combine_state_for_ensemble

mini_batch_size = 32
T = 100

def get_network():
    layers = [MessagePassingLinear(1,20), MessagePassingReLU(), MessagePassingLinear(20,20), MessagePassingReLU(), MessagePassingLinear(20,1)]
    loss = MessagePassingMSE()
    fresh_network = MessagePassingNetwork(layers=layers, loss=loss, learning_rate=0.001)
    return fresh_network


def linspace(start, stop, num):
    return [start + i*(stop-start)/(num-1) for i in range(num) ]

def logspace(start, stop, num):
    return [start * (stop/start)**(i/(num-1)) for i in range(num) ]


def get_task(num_examples, mini_batch_size, random_amplitude):
    if not random_amplitude:
        amp = 1.0
    else:
        amp_range = 1.0, 5.0
        amp = random.random() * (amp_range[1]-amp_range[0]) + amp_range[0]

    phase_range = 0, 3.1415
    phase = random.random() * (phase_range[1]-phase_range[0]) + phase_range[0]

    return SinTask(start=-5, stop=5, num_examples=num_examples, batch_size=mini_batch_size, params={"amp": amp, "phase": phase})

num_runs = 50

data = {}

loss = MessagePassingMSE()

# linspace 0.01 to 0.001 with 10 steps use linspace
for lr in logspace(0.1, 0.001, 20):
    sum = 0
    for i in range(num_runs):


        network = get_network()
        inner_states = network.init_states()

        torch_model, _ = network.totorch(inner_states)


        # adam opt
        optimizer = torch.optim.Adam(torch_model.parameters(), lr=lr)

        task = get_task(mini_batch_size*T+mini_batch_size, mini_batch_size, random_amplitude=True)
        agg_outer = 0
        for t in range(T):
            x, y, _ = task.get_next_train_batch()
            guess = torch_model(x)

            # use  SGDMessageMSELoss to compute loss
            loss_ = loss.forward(guess, y)

            # backwards
            torch_model.zero_grad()
            loss_.backward()

            # update
            optimizer.step()
            agg_outer += loss_.item()


        sum += agg_outer

    # save data
    data[lr] = sum/num_runs
    print("lr: ", lr, " avg loss: ", sum/num_runs)




# plot losses during training
import matplotlib.pyplot as plt

plt.plot(list(data.keys()), list(data.values()))
plt.xlabel("learning rate")
plt.ylabel("loss")

# log space for lr
plt.xscale('log')

plt.savefig('examples/adam_sinewave_lr_randamp.png')
















