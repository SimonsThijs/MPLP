"""
This example shows how to train the MPLP on a set of sinewave functions
"""

import torch
import functorch
from mplp.network import MessagePassingNetwork, unroll_fn
from mplp.models import SGDUpdateFunction, SGDMessageCrossentropyLoss, SGDMessageReLUActivation, SGDMessageLinear, LearningRate
from mplp.layers import MessagePassingLinear, MessagePassingReLU, MessagePassingCrossEntropy
from mplp.tasks import MNISTTask
from mplp.util import list_of_dicts_combine_state_for_ensemble