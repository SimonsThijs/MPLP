import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import math


# Base classes
class MessageBackward:
    def detach(self):
        pass

    def backward(self, receiving_messages, *args, **kwarg):
        """This function takes as input an incoming message and calculates the output message"""
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError

class MessageLayer(MessageBackward):

    def forward(self, input, device):
        """Input takes batches"""
        raise NotImplementedError


def batch_entropy(x):
    """ Estimate the differential entropy by assuming a gaussian distribution of
        values for different samples of a mini-batch.
        
        source: https://github.com/peerdavid/layerwise-batch-entropy/blob/main/experiment_fnn/batch_entropy.py
    """
    if(x.shape[0] <= 1):
        # raise Exception("The batch entropy can only be calculated for |batch| > 1.")
        return 1.0

    x = torch.flatten(x, start_dim=1)
    x_std = torch.std(x, dim=0)
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
    return torch.mean(entropies)

############# Activation functions #############
class MessagePassingActivation(MessageLayer):

    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        return

    def forward(self, inner_state, input, be_reg=False):
        """
        Basically just a regular forward pass, but we also return the intermediate values
        And we calculate the batch entropy if be_reg is set to True

        inner_state:    the state of the layer, this is None for activation layers because activations do not keep states like linear layers do
        input:          the input to the layer [batch_size, N] N is the number of nodes for this layer
        """

        activation, features = self._forward(input) #call the implementation of the activation function
        
        if be_reg: #check if we need to calculate the batch entropy, this saves computation
            be = batch_entropy(activation)
        else:
            be = None

        return activation, features, be

    def _forward(self, input):
        raise NotImplementedError
    
    def init_state(self, n):
        # return empty tensor because activations have no state
        def get_weight():
            return torch.tensor([], device=self.device)

        if not n:
            return get_weight()
        else:
            return torch.stack([get_weight() for _ in range(n)], dim=0)

    def backward(self, receiving_messages, outer_model, outer_state, intermediates):
        """
        This function applies to all different types of activation functions
        This function takes as input an incoming messages and intermediate values from the forward pass, and calculates the output messages

        receiving_messages: the messages that are received by this layer [batch_size, N, message_size] N is the number of nodes for this layer
        intermediates:      the intermediate values that were calculated during the forward pass [batch_size, N, intermediate_size]

        outer_model:        the message generating model for the activation function without state, this is generated using: functorch.combine_state_for_ensemble
        outer_state:        the outer parameters of the model, determine the behaviour of our MPLP
        """

        #input to our message generating functions
        net_input = torch.cat((receiving_messages, intermediates), dim=-1) #[batch_size, N, message_size+intermediate_size]

        # reshape to make it fit the net
        shape = net_input.size()
        batch_size = shape[0]
        N = shape[1] # number of nodes in the activation layer
        net_in_size = shape[-1]

        # fit the input to the network
        net_input = net_input.view(batch_size*N,net_in_size) #[batch_size*N, message_size+intermediate_size] 
        net_output = outer_model(*outer_state, net_input)

        # reshape back
        net_out_size = net_output.size()[-1]
        messages = net_output.view(batch_size,N,net_out_size)
        
        return messages
    
    def is_torch_compatible(self, torch_layer):
        # takes as input a torch layer and returns true if it is compatible with this layer
        # not really used anymore
        return isinstance(torch_layer, type(self.totorch()))
    
    def __str__(self):
        return "{}, {} -> {}".format(type(self).__name__, "x", "x")

class MessagePassingReLU(MessagePassingActivation):

    def _forward(self, input):
        return F.relu(input), input.unsqueeze(-1) #[batch_size, N], [batch_size, N, 1]

    def totorch(self, s=None):
        return nn.ReLU()


class MessagePassingTanh(MessagePassingActivation):

    def _forward(self, input):
        return torch.tanh(input), input.unsqueeze(-1) #[batch_size, N], [batch_size, N, 1]
    
    def totorch(self, s=None):
        return nn.Tanh()


############# Linear layer #############
class MessagePassingLinear(MessageLayer):

    def __init__(self, size_in, size_out, device=torch.device("cpu")):
        self.size_in = size_in # number of input nodes
        self.size_out = size_out # number of output nodes
        self.device = device


    def init_state(self, n):
        """
        This function initializes the inner-state of the layer.
        We use the same initialization as in pytorch from: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        We combine the weights and biases into one tensor to make it easier to work with

        n: the number of initializations to compute
        """

        def random_weights():
            weights = torch.empty(self.size_in+1, self.size_out, device=self.device) #including bias weights

            # weight initialization is the same as in torch
            weights_copy = weights[:-1, :] # exclude bias weights
            nn.init.kaiming_uniform_(weights_copy, a=math.sqrt(5), mode='fan_out') #use fan_out here because our data is not transposed as in pytorch, i dont know exactly why this works but it gives exactly the same outputs as the pytorch implementation at initialization this way
            weights[:-1, :] = weights_copy

            # do bias initialization
            bias = torch.empty(self.size_out, device=self.device)
            bound = 1 / math.sqrt(self.size_in) if self.size_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)
            weights[-1, :] = bias
            return weights #this is just a regular tensor, not a parameter or anything
        
        if not n:
            return random_weights()
        else:
            return torch.stack([random_weights() for _ in range(n)], dim=0)

    def forward(self, inner_state, input):
        """
        inner_state:    the state of the layer, these are the weights of the layer [batch_size, size_in+1, size_out]
        input:          the input to the layer [batch_size, size_in] 
        """

        batch_size = input.size()[0]

        #create the bias weights
        ones = torch.ones(batch_size, 1, device=self.device) 
        input = torch.cat((input, ones), 1)

        result = torch.matmul(input, inner_state) #just a regular linear layer

        return result, input #input is used as the 'intermediate' in the backward

    def backward(self, receiving_messages, outer_model_updatefunc, outer_state_updatefunc, 
                        outer_model_messagelinear, outer_state_messagelinear, intermediate, inner_state, 
                        skip_message_calc=False, use_all_input_features=False):
        """
        The most complicated backwards pass of all the layers because we need to calculate the messages for each weight in the network
        and then combine them to get the output messages

        receiving_messages:         the messages that are received by this layer [batch_size, size_out, message_size]
        intermediate:               the intermediate value of the layer, this is the input of the forward pass [batch_size, size_in+1]
        inner_state:                the weights of the layer that are used in the forward pass [size_in+1, size_out]

        outer_model_updatefunc:     the outer model that is used to update the outer state, without state
        outer_state_updatefunc:     the outer state that is used to update the outer state
        outer_model_messagelinear:  the outer model that is used to calculate the messages, without state
        outer_state_messagelinear:  the outer state that is used to calculate the messages

        skip_message_calc:          if true, the messages are not calculated, this is used for the first layer because no message can be passed backwards so they should also not be calculated
        use_all_input_features:     if true message generating model and weight update model get the same input features
        """

        #getting the dimensions:
        batch_size = intermediate.size()[0]
        n_weights = (self.size_in+1) * self.size_out
        message_size = receiving_messages.size()[-1]

        # reshape the weights of the layer:
        weights_flattened = inner_state.flatten() #[n_weights]
        weights_flattened = torch.unsqueeze(weights_flattened, dim=-1) #[n_weights, 1]
        weights_flattened = weights_flattened.expand((n_weights,batch_size)) #[n_weights, batch_size] this is done to get a copy of the weights for every example in the batch
        weights_flattened = weights_flattened.transpose(1, 0) #[batch_size, n_weights] this step could be avoided by unsqueezing the weights in the first dimension
        weights_flattened = weights_flattened.flatten() #[batch_size*n_weights]
        weights_flattened = torch.unsqueeze(weights_flattened, dim=-1) #[batch_size*n_weights, 1]

        # reshape the intermediate value:
        forward_input = torch.unsqueeze(intermediate, dim=-1) #[batch_size, size_in+1, 1]
        forward_input = forward_input.expand((batch_size,self.size_in+1,self.size_out)) #[batch_size, size_in+1, size_out] every linear node needs access to what was inputted to the network
        forward_input = forward_input.flatten(start_dim=0) #[batch_size*n_weights]
        forward_input = torch.unsqueeze(forward_input, dim=-1) #[batch_size*n_weights, 1]

        receiving_messages = receiving_messages.repeat((1,self.size_in+1,1)) #[batch_size, size_in+1, size_out, message_size] every linear node needs access to the messages that are received
        receiving_messages = receiving_messages.view((batch_size*n_weights, message_size)) #[batch_size*n_weights, message_size]

        
        if use_all_input_features: #message generating model and weight update model get the same input
            net_input_updatefunc = torch.cat((receiving_messages, forward_input, weights_flattened), dim=-1) #[batch_size*n_weights, 2+message_size]
        else:
            # in some cases you dont want to give the same inputs to the message generating model and the weight update model
            # make sure your models can handle this
            net_input_updatefunc = torch.cat((receiving_messages, forward_input), dim=-1) #[batch_size*n_weights, 1+message_size]


        # calculate the weight updates
        weight_updates = outer_model_updatefunc(*outer_state_updatefunc, net_input_updatefunc) #[batch_size*n_weights, 1]
        delta_weights = weight_updates.view(batch_size, self.size_in+1, self.size_out) #[batch_size, size_in+1, size_out] this is how the weights of the model should be updated according to our learned model


        # now we calculate the messages that should be sent to the previous layer

        if skip_message_calc: #if this is the first layer, no message can be passed backwards so we dont calculate them this saves quite some time because the first layer is the largest
            messages = None
        else:

            if use_all_input_features: #message generating model and weight update model get the same input
                net_input_messagelinear = net_input_updatefunc
            else:
                net_input_messagelinear = torch.cat((receiving_messages, weights_flattened), dim=-1) #note that for the messages we need to input the weights


            messages = outer_model_messagelinear(*outer_state_messagelinear, net_input_messagelinear) #[batch_size*n_weights, message_size]

            messages = messages.view(batch_size, n_weights, message_size) #[batch_size, n_weights, message_size]
            messages = messages.view(batch_size, int(n_weights/self.size_out), self.size_out, message_size) #[batch_size, size_in+1, size_out, message_size] note: all size_in+1 nodes receive size_out messages so we still have to combine them


            # reduce the messages:
            # messages = messages.mean(dim=2) we suspect mean would generalize better to larger architectures but sum is more related to SGD see appendix of my thesis
            messages = messages.sum(dim=2) #[batch_size, size_in+1, message_size] now we have a message that can be sent to the previous layer for each in-node

            messages = messages[:, :-1, :] #remove the bias messages

        return messages, delta_weights

    def update_params(self, learning_rate, inner_state, weight_updates):
        """
        Use the calculated weight updates from the backwards pass to update the weights of the layer

        inner_state:    the current weights of the layer [size_in+1, size_out]
        weight_updates: the calculated weight updates [size_in+1, size_out]

        learning_rate:  scales the weight updates, can be a learned parameter or a constant
        """


        if inner_state is not None:
            inner_state = inner_state - torch.clamp(learning_rate * weight_updates.mean(dim=0), -1, 1) #scale and clamp the weight updates
        else:
            raise Exception("Delta weights are None try running backwards first")
        
        return inner_state
    

    # torch conversion functions
    def totorch(self, state):
        weights = state
        layer = nn.Linear(self.size_in, self.size_out)
        weights_detached = weights.detach()
        layer.weight = torch.nn.Parameter(torch.transpose(weights_detached[:-1, :], 0, 1))
        layer.bias = torch.nn.Parameter(torch.reshape(weights_detached[-1, :], (self.size_out,)))
        return layer
    
    def is_torch_compatible(self, torch_layer):
        return isinstance(torch_layer, nn.Linear) and torch_layer.in_features == self.size_in and torch_layer.out_features == self.size_out

    def set_torch_weights(self, torch_layer):
        # set the weights of this layer to the weights of the torch layer
        with torch.no_grad():
            self.weights[:-1, :] = torch.transpose(torch_layer.weight.data.clone().detach(), 0, 1)
            self.weights[-1, :] = torch.reshape(torch_layer.bias.data.clone().detach(), (self.size_out,))

    def __str__(self):
        return "{}, {} -> {}".format(type(self).__name__, self.size_in, self.size_out)




############# Loss layers #############
class MessagePassingLoss(MessageBackward):

    def _backward(self, receiving_messages, y, outer_model, outer_state, intermediates):
        """
        This function can be used by all different types of loss layers

        receiving_messages: the messages that are received by the loss layer these messages to the loss layer are different from the messages from the other layers [batch_size, N]
                             note that the size of the messages are of dimenionality 1 
        y:                  the target values [batch_size, N] N is the number of nodes in this layer
        intermediates:      the intermediate values that were calculated by the forward pass of the model [batch_size, N, n_intermediates] 
                             n_intermediates can be bigger than 1 in the case of the cross-entropy loss
    
        outer_model:        the model that calculates the messages for the loss layer, without state
        outer_state:        the state/parameters of the model that calculates the messages for the loss layer
        """


        n_output_nodes = intermediates.size()[-2]
        batch_size = intermediates.size()[0]

        receiving_messages = torch.unsqueeze(receiving_messages, dim=-1) # [batch_size, N, 1]

        y = torch.unsqueeze(y, dim=-1) # [batch_size, N, 1]

        net_input = torch.cat((receiving_messages, intermediates, y), dim=-1) # [batch_size, N, 1+n_intermediates+1]
        
        # reshape to fit the model
        net_in_size = net_input.size()[-1] # 1+n_intermediates+1
        net_input = net_input.view(batch_size*n_output_nodes, net_in_size) # [batch_size*N, 1+n_intermediates+1]

        # calculate the messages
        net_output = outer_model(*outer_state, net_input) # [batch_size*N, message_size]

        # reshape output messages back
        net_out_size = net_output.size()[-1] # message_size
        messages = net_output.view(batch_size, n_output_nodes, net_out_size) # [batch_size, N, message_size]

        return messages
    
    def totorch(self):
        return self.forward

    def __str__(self):
        return type(self).__name__



class MessagePassingMSE(MessagePassingLoss):

    @staticmethod
    def forward(guess, y, reduction='mean', return_features=False):
        """
        This function implements the mean squared error loss, uses pytorch implementation

        guess:              the output of the model [batch_size, N] N is the number of nodes in the layer, 1 for e.g., final layer of sinewave
        y:                  the target values [batch_size]

        reduction:          same as from pytorch
        return_features:    return the intermediates that are used in the backwards pass
        """

        if return_features:
            return F.mse_loss(guess, y, reduction=reduction)/2, guess.unsqueeze(-1)
        else:
            return F.mse_loss(guess, y, reduction=reduction)/2
        

    def backward(self, receiving_messages, y, outer_model, outer_state, intermediates):
        # calls _backwards and should just work in crossentropy loss we need to do some extra calculations
        return self._backward(receiving_messages, y, outer_model, outer_state, intermediates)



    # def calculate(self, guess, y, reduction='none'):
    #     # TODO: get rid of this calculate function
    #     return torch.squeeze(self.loss(guess, y, reduction))
    



class MessagePassingCrossEntropy(MessagePassingLoss):
    # this is the softmax and the crossentropy combined just as with F.cross_entropy

    
    @staticmethod
    def forward(guess, y, reduction="mean", return_features=False):
        """
        This function implements the crossentropy loss, own implementation

        guess:              the output of the model [batch_size, N] N is the number of nodes in the layer, 10 for e.g., final layer of MNIST
        y:                  the target values [batch_size]

        reduction:          same as from pytorch
        return_features:    return the intermediates that are used in the backwards pass
        """

        #cross entropy is invariant to offsets in the input, so we subtract the max value from the input
        translated_inputs = guess - torch.max(guess, dim=-1, keepdim=True)[0] # [batch_size, N]

        exp_nom = torch.exp(translated_inputs) # [batch_size, N]
        exp_denom = torch.sum(exp_nom, axis=-1) # [batch_size]

        # repeat the denominator for the same number of times as the last dim of the nominator
        # this is done to make the division work
        exp_denom = torch.unsqueeze(exp_denom, dim=-1) # [batch_size, 1]
        exp_denom = exp_denom.repeat(1,exp_nom.shape[-1]) # [batch_size, N]

        #softmax:
        result = exp_nom / exp_denom # [batch_size, N]
        probs = torch.clip(result, 1e-8, 1-1e-8) # keep between 1e-8 and 1-1e-8 might make optimization easier

        n_output_nodes = probs.size()[-1] # N,    10 in case of MNIST
        y = F.one_hot(y, num_classes=n_output_nodes) # [batch_size, N]

        # cross entropy:
        log_in = torch.log(probs) # [batch_size, N]
        result = -(y * log_in) # [batch_size, N]


        # reduce the batch AND the nodes
        if reduction == "mean":
            result = result.sum(dim=-1)
            result = result.mean(dim=-1)
        elif reduction == "sum":
            result = result.sum(dim=-1)
            result = result.sum(dim=-1)
        elif reduction == "none":
            pass
        else:
            raise Exception("Invalid reduction type")
        
        if return_features:
            # instead of the inputs we return the probabilities, see appendix of thesis about the cross-entropy
            # we might want to add the inputs as well
            features = torch.cat((probs.unsqueeze(-1),), dim=-1)
            return result, features # [batch_size, N], [batch_size, N, 1] can depend on reduction
        
        return result 
    

    def backward(self, receiving_messages, y, outer_model, outer_state, intermediates):
        n_output_nodes = intermediates.size()[-2]
        y = F.one_hot(y, num_classes=n_output_nodes) # [batch_size, N] convert to one hot
        return self._backward(receiving_messages, y, outer_model, outer_state, intermediates)
    


    # def calculate(self, guess, y, reduction='none'):
    #     return self.loss(guess, y, reduction=reduction)
    


