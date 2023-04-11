# Message Passing Learning Protocol
Reimplementation of the Message Passing Learning Protocol framework for my master's thesis. This repository should be a fairly clean and simple implementation of the mplp framework and can be a good starting point for people to better understand expressive learned optimizer and bilevel optimization in general. The code for the framework can be found in the 'mplp/' folder and contains many comments.

Main contribution of my thesis is the addition of batch entropy regularization which can make the training of mplp like methods significantly more predictable.

Master thesis: \
Original paper: https://arxiv.org/abs/2007.00970 


### Installation

```
git clone https://github.com/SimonsThijs/MPLP.git
cd mplp
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### Examples

Examples can be found in the examples folder. Weights and Biases is turned on by default so if wandb is not configured make sure you turn it off with the --wandb argument.

Train an mplp on the sinewave task. This example is the most basic and cleanest demonstration of the mplp framework:
```
python3 examples/mplp_sinewave.py --wandb False
```

Train an mplp on the mnist problem using batch entropy regularization as described in my thesis:
```
python3 examples/mplp_mnist.py --wandb False
```

Mimic SGD using the MPLP framework:
```
python3 examples/sgd_mnist.py
```


### Differences with the original
This implementation sometimes deviates from the original, the most important difference is that we have decided to not use RNNs in the linear nodes. This is done to make the understanding of the MPLP more easy. Start simple and gradually extend is the philosphy used.
