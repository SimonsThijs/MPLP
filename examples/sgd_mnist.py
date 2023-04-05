"""
This examples shows how to use the MPLP framework as a simple SGD optimizer.
The trick is in the models that we used from mplp.models
The thesis explains how we have found these models see Section Expressiveness to imitate SGD
"""

import os
from mplp.network import MessagePassingNetwork, unroll_fn
from mplp.models import SGDUpdateFunction, SGDMessageCrossentropyLoss, SGDMessageReLUActivation, SGDMessageLinear, LearningRate
from mplp.layers import MessagePassingLinear, MessagePassingReLU, MessagePassingCrossEntropy
from mplp.tasks import MNISTTask
from mplp.util import list_of_dicts_combine_state_for_ensemble

mini_batch_size = 32
T = 300

# optimizee
layers = [MessagePassingLinear(size_in=784, size_out=100), MessagePassingReLU(), MessagePassingLinear(size_in=100, size_out=10)]
loss = MessagePassingCrossEntropy()
network = MessagePassingNetwork(layers, loss)
inner_states = network.init_states()

#optimizer
model = {
    'updatefunc': SGDUpdateFunction(),
    'messageloss': SGDMessageCrossentropyLoss(),
    'messagelinear': SGDMessageLinear(),
    'messagerelu': SGDMessageReLUActivation(),
    'lr': LearningRate(0.01),
}
outer_model, outer_states = list_of_dicts_combine_state_for_ensemble([model,])

# task
task = MNISTTask(batch_size=mini_batch_size)




# in the unrolling we do the forward and the backward passes
outer_loss, new_inner_states, loss_list = unroll_fn(inner_states, outer_states, T, network, outer_model, task, return_loss_list=True)





# plot losses during training
import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.xlabel('t')
plt.ylabel('loss')
plt.title('MNIST SGD using MPLP framework')
folder_of_this_file = os.path.dirname(os.path.abspath(__file__))
plt.savefig(folder_of_this_file + '/sgd_mnist.png')
















