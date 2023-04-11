"""
This example shows how to train the MPLP on a set of sinewave functions
This is the most basic example of how to train the MPLP
Most of this code is initializing optimizee, optimizer, tasks etc.
The important part is the function at the end of this file called 'meta_learn'
"""
import math
import argparse
import random
import torch
import os

from mplp.network import MessagePassingNetwork, unroll_fn
from mplp.models import LearningRate, UpdateFunction, MessageLinear, MessageRelu, MessageLoss, SGDMessageCrossentropyLoss, SGDMessageLinear, SGDMessageReLUActivation
from mplp.layers import MessagePassingLinear, MessagePassingReLU, MessagePassingMSE
from mplp.tasks import SinTask
from mplp.util import *
from mplp.gradient_calculations import analytical_gradient

from util import get_device, save_results, list_of_lists_to_tensor

parser = argparse.ArgumentParser(
    description='Learning an optimizer on a sinewave')
parser.add_argument(
    '--message_passing',
    type=str,
    default='True',
    choices=['True', 'False'],
    help='Use message passing or only learn the update function')

parser.add_argument(
    '--hidden_size',
    type=int,
    default=50,
    help='Number of nodes in the hidden layer of our optimizer')

parser.add_argument(
    '--T',
    type=int,
    default=200,
    help='Number of inner-update steps that are done on the sinewave')

parser.add_argument(
    '--K',
    type=int,
    default=10,
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
    default=1,
    help='Message size of our message passing neural networks only used if message_passing is True')

parser.add_argument(
    '--inner_batch_size',
    type=int,
    default=32,
    help='Inner mini batch size')

parser.add_argument(
    '--n_validation_tasks',
    type=int,
    default=5,
    help='On how many tasks to validate')

parser.add_argument(
    '--result_path',
    type=str,
    default='mplp_sinewave/',
    help='Path where to save the results')

parser.add_argument(
    '--device',
    type=str,
    default=None,
    help='which device to use if not specified the device will be chosen automatically')

parser.add_argument(
    '--validate_every',
    type=int,
    default=20,
    help='how often to run the meta validation')

parser.add_argument(
    '--wandb',
    type=str,
    default='True',
    choices=['True', 'False'],
    help='Use wandb to log the results')


run_name = str(random.randint(0, 1e10)) # used for saving the results somewhere
folder_of_this_file = os.path.dirname(os.path.abspath(__file__))

args = parser.parse_args()
args.message_passing = args.message_passing == 'True'
args.wandb = args.wandb == 'True'

if args.wandb:
    import wandb
    wandb.init(
        project="mplp_sinewave",
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
######################      begin code      ######################


# optimizee
def get_network():
    layers = [MessagePassingLinear(1,20, device), MessagePassingReLU(device=device), MessagePassingLinear(20,20, device), MessagePassingReLU(device=device), MessagePassingLinear(20,1, device)]
    loss = MessagePassingMSE()
    fresh_network = MessagePassingNetwork(layers=layers, loss=loss, device=device, learning_rate=0.001)
    return fresh_network

def init_fn():
    network = get_network()
    inner_states = network.init_states()
    return network, inner_states




# inner-optimizer - mplp
def get_model():
    if args.message_passing:
        model = {
            'updatefunc': UpdateFunction(args.message_size, args.hidden_size, device=device),
            'messageloss': MessageLoss(args.message_size, args.hidden_size, device=device),
            'messagelinear': MessageLinear(args.message_size, args.hidden_size, device=device),
            'messagerelu': MessageRelu(args.message_size, args.hidden_size, device=device),
        }
    else:
        # assert message_size == 1
        if args.message_size != 1:
            raise ValueError("message_size must be 1 if message_passing is False")
        
        model = {
            'updatefunc': UpdateFunction(args.message_size, args.hidden_size, device=device),
            'messageloss': SGDMessageCrossentropyLoss(device=device),
            'messagelinear': SGDMessageLinear(device=device),
            'messagerelu': SGDMessageReLUActivation(device=device),
        }
    return model

outer_model, outer_states = list_of_dicts_combine_state_for_ensemble([get_model(),])
outer_states = apply_to_nested(lambda x: x[0], outer_states) # ugly but list_of_dicts_combine_state_for_ensemble only supports returning a dict of lists 


# tasks
def get_task(num_examples, inner_batch_size, device, random_amplitude):
    if not random_amplitude:
        amp = 1.0
    else:
        amp_range = 1.0, 5.0
        amp = random.random() * (amp_range[1]-amp_range[0]) + amp_range[0]

    phase_range = 0, 3.1415
    phase = random.random() * (phase_range[1]-phase_range[0]) + phase_range[0]

    return SinTask(start=-5, stop=5, num_examples=num_examples, batch_size=inner_batch_size, device=device, params={"amp": amp, "phase": phase})


# validation stuff
val_tasks = [get_task(args.T*args.inner_batch_size+args.inner_batch_size, args.inner_batch_size, random_amplitude=True, device=device) for _ in range(args.n_validation_tasks)]
val_network = get_network()
val_inner_states = [val_network.init_states() for _ in range(args.n_validation_tasks)]

def validate(outer_states, val_inner_state, val_task):
    # reset the val task
    val_task.index = 0
    # unroll the network
    loss, new_inner_state, final_loss = unroll_fn(val_inner_state, outer_states, args.T, val_network, outer_model, val_task, return_final_loss=True)
    return loss, final_loss

def validate_(outer_state):
    val_losses = []
    val_final_losses = []
    for i in range(args.n_validation_tasks):
        loss, final_loss = validate(outer_state, val_inner_states[i], val_tasks[i])
        val_losses.append(loss.item())
        val_final_losses.append(final_loss.item())
    
    return {'loss': np.array(val_losses), 'final_loss': np.array(val_final_losses)}
    





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

    while meta_step < args.n_meta_steps:

        if t + args.K > args.T:
            t = 0
            # we sample a new task and network
            network, inner_states = init_fn()
            task = get_task(args.T*args.inner_batch_size+args.inner_batch_size, args.inner_batch_size, random_amplitude=True, device=device)

        # we calculate the gradient and update the outer model
        grad, inner_states = analytical_gradient(unroll_fn, args.K, inner_states, outer_states, network, outer_model, task)

        # we update the outer model with the gradients
        outer_states = apply_to_three_nested(update, optimizer_state, outer_states, grad)

        if meta_step % args.validate_every == 0:
            val_data = validate_(outer_states)
            log({
                "step n": meta_step, 
                "loss": val_data['loss'].mean(), 
                "final_loss": val_data['final_loss'].mean()
                 })
            losses_history.append(val_data)


        meta_step += 1
        t += args.K

    return losses_history, outer_states


losses_history, outer_states = meta_learn(outer_states)

save_results([losses_history, outer_states], folder_of_this_file + args.result_path, experiment_name, run_name)
print("finished", experiment_name)































