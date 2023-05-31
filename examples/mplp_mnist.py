"""
This example shows how to train the MPLP on the mnist dataset
This is the most basic example of how to train the MPLP
Most of this code is initializing optimizee, optimizer, tasks etc.
The important part is the function at the end of this file called 'meta_learn'
"""
import math
import argparse
import random
import torch
import torch.nn as nn
import os
import numpy as np
import time

from mplp.network import MessagePassingNetwork, unroll_fn
from mplp.models import LearningRate, UpdateFunction, MessageLinear, MessageRelu, MessageLoss
from mplp.layers import MessagePassingLinear, MessagePassingReLU, MessagePassingCrossEntropy
from mplp.tasks import MNISTTask, FashionMNISTTask
from mplp.util import *
from mplp.gradient_calculations import analytical_gradient

from util import get_device, save_results, list_of_lists_to_tensor

parser = argparse.ArgumentParser(
    description='Learning an optimizer on a sinewave')

parser.add_argument(
    '--hidden_size',
    type=int,
    default=50,
    help='Number of nodes in the hidden layer of our optimizer')

parser.add_argument(
    '--T',
    type=int,
    default=100,
    help='Number of inner-update steps that are done on the sinewave')

parser.add_argument(
    '--K',
    type=int,
    default=5,
    help='Unroll length')

parser.add_argument(
    '--lr',
    type=float,
    default=1e-4,
    help='outer-learning rate')

parser.add_argument(
    '--b1',
    type=float,
    default=0.9,
    help='Adam beta 1')

parser.add_argument(
    '--b2',
    type=float,
    default=0.999,
    help='Adam beta 2')

parser.add_argument(
    '--n_meta_steps',
    type=int,
    default=2e5,
    help='Number of meta steps to be done')

parser.add_argument(
    '--message_size',
    type=int,
    default=24,
    help='Message size of our message passing neural networks only used if message_passing is True')

parser.add_argument(
    '--inner_batch_size',
    type=int,
    default=8,
    help='Inner mini batch size')

parser.add_argument(
    '--result_path',
    type=str,
    default='mplp_mnist/',
    help='Path where to save the results')

parser.add_argument(
    '--device',
    type=str,
    default=None,
    help='which device to use if not specified the device will be chosen automatically')

parser.add_argument(
    '--validate_every',
    type=int,
    default=200,
    help='how often to run the meta validation')

parser.add_argument(
    '--be_reg',
    type=str,
    default='True',
    choices=['True', 'False'],
    help='Use batch entropy regularization')

parser.add_argument(
    '--learn_lr',
    type=str,
    default='True',
    choices=['True', 'False'],
    help='Learn the learning rate')

parser.add_argument(
    '--n_validation_inits',
    type=int,
    default=2,
    help='Number of validation inits')

parser.add_argument(
    '--wandb',
    type=str,
    default='True',
    choices=['True', 'False'],
    help='Use wandb to log the results')


run_name = str(random.randint(0, 1e10)) # used for saving the results somewhere
folder_of_this_file = os.path.dirname(os.path.abspath(__file__))

args = parser.parse_args()
args.wandb = args.wandb == 'True'
args.be_reg = args.be_reg == 'True'

if args.wandb:
    import wandb
    wandb.init(
        project="mplp_mnist",
        config=dict(vars(args)) #this is nice because it saves all the arguments in wandb
    )

log_buffer = {}
def log(data, commit=True):
    if args.wandb:
        wandb.log(data, commit=commit)
    else:
        log_buffer.update(data)
        if commit:
            print(log_buffer)
            log_buffer.clear()

def get_experiment_name():
    args_dict = dict(vars(args))
    args_dict.pop('result_path')
    args_dict.pop('device')
    experiment_name = " ".join([f'{key}:{value}' for key, value in sorted(args_dict.items())])
    return experiment_name

experiment_name = get_experiment_name()

if args.device:
    device = torch.device(args.device)
else:
    device = get_device()

print("device being used:", device)



######################      end config      ######################
######################     begin models     ######################


class BatchNormalize(nn.Module):
    def __init__(self, input_size):
        super(BatchNormalize, self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        # buffer for the running mean and std
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_std', torch.ones(input_size))

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        std = torch.std(x, dim=0, keepdim=True) + 1e-8

        momentum = 1.0 # this means the buffers are overwritten at each call, there are no running stats
        # update the running mean and std
        self.running_mean = momentum*mean + (1.0-momentum)*self.running_mean
        self.running_std = momentum*std + (1.0-momentum)*self.running_std

        # normalize the input
        x = (x - self.running_mean) / self.running_std

        # scale and shift
        x = self.gamma * x + self.beta

        return x

class MessageLoss(nn.Module):
    def __init__(self, message_size, hidden_size, input_size=3, device='cpu'):
        super(MessageLoss, self).__init__()
        self.func = nn.Sequential(
            BatchNormalize(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x) 


class MessageRelu(nn.Module):
    def __init__(self, message_size, hidden_size, device='cpu'):
        super(MessageRelu, self).__init__()
        self.func = nn.Sequential(
            BatchNormalize(message_size+1),
            nn.Linear(message_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x) 


class MessageLinear(nn.Module):
    def __init__(self, message_size, hidden_size, device='cpu'):
        super(MessageLinear, self).__init__()
        self.func = nn.Sequential(
            BatchNormalize(message_size+1),
            nn.Linear(message_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x) 

class UpdateFunction(nn.Module):
    def __init__(self, message_size, hidden_size, output_size=1, device='cpu'):
        super(UpdateFunction, self).__init__()
        self.func = nn.Sequential(
            BatchNormalize(message_size+1),
            nn.Linear(message_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x)


######################      end models      ######################
######################      begin code      ######################



# optimizee
def get_network():
    layers = [MessagePassingLinear(28*28,32, device), 
              MessagePassingReLU(device=device), 
              MessagePassingLinear(32,20, device), 
              MessagePassingReLU(device=device), 
              MessagePassingLinear(20,10, device)]
    loss = MessagePassingCrossEntropy() 
    fresh_network = MessagePassingNetwork(layers=layers, loss=loss, device=device, learning_rate=0.01, be_reg=args.be_reg)
    return fresh_network
def init_fn():
    network = get_network()
    inner_states = network.init_states()
    return network, inner_states


# inner-optimizer - mplp
def get_model():
    model = {
        'updatefunc': UpdateFunction(args.message_size, args.hidden_size, device=device),
        'messageloss': MessageLoss(args.message_size, args.hidden_size, device=device),
        'messagelinear': MessageLinear(args.message_size, args.hidden_size, device=device),
        'messagerelu': MessageRelu(args.message_size, args.hidden_size, device=device),
    }

    if args.learn_lr:
        model['lr'] = LearningRate(0.01, device=device)

    return model

outer_model, outer_states = list_of_dicts_combine_state_for_ensemble([get_model(),])
outer_states = apply_to_nested(lambda x: x[0], outer_states) # ugly but list_of_dicts_combine_state_for_ensemble only supports returning a dict of lists 


# validate on the same network and the same task data
val_tasks = [MNISTTask(batch_size=args.inner_batch_size, device=device) for _ in range(0)]
val_network = get_network()
val_inner_states = [val_network.init_states() for _ in range(args.n_validation_inits)]


mnist_task = MNISTTask(args.inner_batch_size, device=device)
fashionmnist_task = FashionMNISTTask(args.inner_batch_size, device=device)

validation_tasks = [mnist_task, fashionmnist_task]

# returns list of losses for each task, mean of the different inits
@torch.no_grad()
def validate(outer_states): 
    
    losses = np.zeros((len(validation_tasks), args.n_validation_inits, 4))
    for i, task in enumerate(validation_tasks):
        old_index = task.index
        task.index = 0 #reset the task index to 0 so we are always validating on the beginning of the dataset
        for j in range(args.n_validation_inits):
            loss, new_inner_state, final_loss, plain_loss, be_loss = unroll_fn(val_inner_states[j], outer_states, args.T, val_network, outer_model, task, return_final_loss=True, return_loss_components=True)
            

            def tensor_to_float(loss): return float(loss.cpu().detach())
            losses[i,j,0] = tensor_to_float(loss)
            losses[i,j,1] = tensor_to_float(final_loss)
            losses[i,j,2] = tensor_to_float(plain_loss)
            losses[i,j,3] = tensor_to_float(be_loss)
            
    
        task.index = old_index
    
    return losses



# implementation of adam, this is the outer optimizer, the optimizer that optimizes the MPLP
class OptState: #for adam
    def __init__(self, first_moment, second_moment):
        self.first_moment = first_moment 
        self.second_moment = second_moment

def init_opt_state(outer_state):
    return OptState(torch.zeros_like(outer_state), torch.zeros_like(outer_state))

def update(optimizer_state, outer_state, grad):

    grad = torch.clip(grad, -1.0, 1.0)

    optimizer_state.first_moment = args.b1*optimizer_state.first_moment + (1-args.b1) * grad
    optimizer_state.second_moment = args.b2*optimizer_state.second_moment + (1-args.b2) * (grad**2)

    update = args.lr * optimizer_state.first_moment / (torch.sqrt(optimizer_state.second_moment) + 1e-11)

    updated = outer_state - update
    return updated




# this is where the magic happens
# same as algorithm 1 in the thesis
def meta_learn(outer_states):

    t = math.inf #inf to begin sampling a new optimizee
    meta_step = 0
    losses_history = []

    optimizer_state = apply_to_nested(init_opt_state, outer_states) # adam optimizer state init
    
    task = mnist_task

    while meta_step < args.n_meta_steps:

        if t + args.K > args.T:
            t = 0
            network, inner_states = init_fn()

        # we calculate the gradient and update the outer model
        grad, inner_states = analytical_gradient(unroll_fn, args.K, inner_states, outer_states, network, outer_model, task)

        # we update the outer model with the gradients
        outer_states = apply_to_three_nested(update, optimizer_state, outer_states, grad)

        if meta_step % args.validate_every == 0:
            # validate
            losses = validate(outer_states)
            losses_history.append(losses)
            losses_means = np.mean(losses, axis=1)
            log({"loss_mnist": losses_means[0,0], "loss_fashionmnist": losses_means[1,0], "final loss mnist": losses_means[0,1],
                          "final loss fashionmnist": losses_means[1,1], "plain loss mnist": losses_means[0,2], "plain loss fashionmnist": losses_means[1,2],
                            "be loss mnist": losses_means[0,3], "be loss fashionmnist": losses_means[1,3]})


        meta_step += 1
        t += args.K

    return losses_history, outer_states


losses_history, outer_states = meta_learn(outer_states)

save_results([losses_history, outer_states], folder_of_this_file + args.result_path, experiment_name, run_name)
print("finished", experiment_name)































