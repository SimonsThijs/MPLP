"""
This examples shows how to use the MPLP framework as a simple SGD optimizer.
The trick is in the models that we used from mplp.models
The thesis explains how we have found these models see Section Expressiveness to imitate SGD
"""


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

# linspace 0.01 to 0.001 with 10 steps use linspace
for lr in logspace(10, 0.001, 20):
    sum = 0
    for i in range(num_runs):
        #optimizer
        model = {
            'updatefunc': SGDUpdateFunction(),
            'messageloss': SGDMessageMSELoss(),
            'messagelinear': SGDMessageLinear(),
            'messagerelu': SGDMessageReLUActivation(),
            'lr': LearningRate(0.01),
        }
        outer_model, outer_states = list_of_dicts_combine_state_for_ensemble([model,])


        network = get_network()
        inner_states = network.init_states()
        task = get_task(mini_batch_size*T+mini_batch_size, mini_batch_size, random_amplitude=True)
        outer_loss, new_inner_states, loss_list = unroll_fn(inner_states, outer_states, T, network, outer_model, task, return_loss_list=True)

        sum += outer_loss.item()

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

folder_of_this_file = os.path.dirname(os.path.abspath(__file__))
plt.savefig(folder_of_this_file + '/sgd_sinewave_lr_randamp.png')

















