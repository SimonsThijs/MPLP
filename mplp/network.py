import torch
import torch.nn as nn

from layers import *

class MessagePassingNetwork():
    """
    Resembles the optimizee.
    Uses the layers from layers.py to build a network.
    """

    def __init__(self, layers, loss, learning_rate=0.001, be_reg=False, device=torch.device("cpu")):
        """
        layers:         list of layers
        loss:           loss layer that was used
        learning_rate:  either a float or a torch.Parameter
        be_reg:         boolean, if true, the network will use the be regularization
        """


        self.layers = layers
        self.loss = loss
        self.device = device
        self.number_of_steps_taken = 0
        self.learning_rate = learning_rate
        self.be_reg = be_reg
    

    def init_states(self, n=None):
        return [l.init_state(n) for l in self.layers] #linear is the only one with a state, namely the weights
    
    def calculate_loss(self, guess, y):
        return self.loss.forward(guess, y, reduction='mean')

    def detach_weights(self):
        """
        Detaching is done to prevent pytorch from constructing a computational graph.
        """
        for l in self.layers:
            l.detach()
        
    def forward(self, inner_states, x):
        """
        This function calls forward on all the layers in the network sequentially.
        It gathers all the intermediates to be used in the backwards pass
        It also calculates the batch entropy regularization if be_reg is true, if not true it will return 0

        inner_states:   list of states for the layers, contains the weights for the linear layer
        x:              input to the network
        """

        intermediates = []

        alpha = 0.5
        be_sum = 0
        be_count = 0
        for l, inner_state in zip(self.layers, inner_states):

            # if activation layer we need to calculate the batch entropy:
            if isinstance(l, MessagePassingActivation):
                be_count += 1
                x, intermediate, be = l.forward(inner_state, x, be_reg=self.be_reg)

                # be is none when be_reg is set to false
                if be is not None and be < alpha:
                    be_sum += alpha-be #equation 11 from the thesis
            else:
                x, intermediate = l.forward(inner_state, x)


            intermediates.append(intermediate)
        
        be = be_sum/be_count #calculate the mean over the layers, also equation 11 from the thesis
        return x, intermediates, be
    
    def backward(self, guess, y, outer_model, outer_state, inner_states, intermediates):
        """
        This function calls backward on all the layers in the network sequentially.
        It gathers all the weight updates to be used in the step function and it passes the message to the next layers

        guess:          output of the network [batch_size, output_size]
        y:              target [batch_size]
        outer_model:    dict of the outer model, contains the models for the layers but not the states
        outer_state:    dict of the outer states, contains the states for the layers aka the outer-parameters, meta-parameters or outer-state depending on who you ask
        inner_states:   list of the inner states, contains the weights for the optimizee aka the inner-parameters or inner-state
        intermediates:  list of the intermediates, contains data that was calculated during the forward pass by the network. 
                         this data is input to the message generating functions and the update functions
        """
        m_0, intermediate = self.loss.forward(guess, y, reduction='none', return_features=True) #initial message is the loss, intermediate is either guess or probs depending on the loss

        m = self.loss.backward(m_0, y, outer_model['messageloss'], outer_state['messageloss'], intermediate) # guess is the intermediate for the loss layer

        weight_updates = []
        # reverse the order because we are going backwards
        for l, intermediate, inner_state in reversed(list(zip(self.layers, intermediates, inner_states))):
            delta_weights = torch.tensor([])

            # this if else basically calls the backwards function with the right model and its state
            if isinstance(l, MessagePassingReLU):
                m = l.backward(m, outer_model['messagerelu'], outer_state['messagerelu'], intermediate)
            elif isinstance(l, MessagePassingTanh):
                m = l.backward(m, outer_model['messagetanh'], outer_state['messagetanh'], intermediate)
            elif isinstance(l, MessagePassingLinear):
                m, delta_weights = l.backward(m, outer_model['updatefunc'], outer_state['updatefunc'], outer_model['messagelinear'], outer_state['messagelinear'], intermediate, inner_state, 
                                                skip_message_calc=(l==self.layers[0])) #if we are at the first layer we do not need to calculate the message
            else:
                raise Exception("This layer is not compatible")

            weight_updates.append(delta_weights)

        weight_updates.reverse()  
        return weight_updates


    def step(self, inner_states, weight_updates, outer_state):
        """
        This function calls 'step' on all the layers in the network sequentially.
        It is mostly a single for loop.

        inner_states:   list of the inner states, contains the weights for the optimizee that are going to be updated
        weight_updates: list of the weight updates, contains the updates for the weights for the optimizee
        outer_state:    dict of the outer states, contains the learned learning rate if it is enabled
        """

        new_inner_states = []

        # check if the learning rate is learned, if so we use it
        if 'lr' in outer_state:
            self.learning_rate = outer_state['lr'][0][0]

        # call update on all linear layers
        for l, weight_update, inner_state in zip(self.layers, weight_updates, inner_states):
            if isinstance(l, MessagePassingLinear):
                updated_inner_state = l.update_params(self.learning_rate, inner_state, weight_update)
            else:
                updated_inner_state = inner_state #non linear layers do not have a state and thus there is also no update
            
            new_inner_states.append(updated_inner_state)


        self.number_of_steps_taken+=1
        return new_inner_states


    def __str__(self):
        result_string = ""
        for l in self.layers:
            result_string += l.__str__()
            result_string += "\n"
        result_string += self.loss.__str__()
        return result_string



    ############################## TORCH utiles ########################################
    def totorch(self, inner_states):
        torch_layers = []
        for l, s in zip(self.layers, inner_states):
            torch_layers.append(l.totorch(s))

        torch_module = nn.Sequential(*torch_layers)

        return torch_module, self.loss.totorch()



def unroll_fn(inner_states, outer_state, K, inner_model, outer_model, task, return_final_loss=False, return_loss_list=False, return_loss_components=False):
    """
    This function is the unroll function. It calls forward and backward K times and returns the loss and the weight updates.
    See Algorithm 2 from the thesis

    inner_states:           list of the inner states, contains the weights for the optimizee aka the inner-parameters or inner-state
    outer_state:            dict of the outer states, contains the states for the layers aka the outer-parameters, meta-parameters or outer-state depending on who you ask
    K:                      number of times the forward and backward pass is called
    inner_model:            this is the network that is optimized, it is a MessagePassingNetwork
    outer_model:            dict of the outer model, contains the models for the layers but not the states
    task:                   the task that is used to get the data
    return_final_loss:      if true the final loss is returned
    return_loss_list:       if true the loss list is returned
    return_loss_components: if true the outer loss and the batch entropy loss are also returned seperately

    Returns the loss and the new inner_state but depending on the config also the loss list and the loss components can be returned

    """

    network = inner_model
    
    # used to keep track of information
    loss_accum = None
    be_accum = 0
    be_count = 0

    loss_list = []

    # K is the truncation length
    for i in range(K):
        x_batch, y_batch, _ = task.get_next_train_batch()

        #run the forward 
        guess, intermediates, be_sum = network.forward(inner_states, x_batch)
        loss = network.calculate_loss(guess, y_batch) 

        # accumulate the loss and the batch entropy
        be_accum += be_sum
        be_count += 1
        if loss_accum is None and i > 0:
            loss_accum = loss
        elif i > 0:
            loss_accum += loss

        #run the backward
        weight_updates = network.backward(guess, y_batch, outer_model, outer_state, inner_states, intermediates) # we also need the intermediates to do the backward pass
        inner_states = network.step(inner_states, weight_updates, outer_state) # here we get the updated inner states


        if return_loss_list:
            loss_list.append(loss.item())



    be_accum /= be_count #keep it between 0 and 0.5
    be_accum *= 2 #keep it between 0 and 1
    be_accum *= 1 * loss.detach() #weight it to be the same as a single loss
    

    #returning the data
    to_return = [loss_accum+be_accum, inner_states] # we always return the loss and the inner states

    if return_loss_list:
        to_return.append(log_list)

    if return_final_loss:
        to_return.append(loss)

    if return_loss_components:
        to_return.append(loss_accum)
        to_return.append(be_accum)

    return tuple(to_return)


















def unroll_fn_rl(inner_states, outer_state, K, inner_model, outer_model, task,
                return_final_loss=False, return_loss_list=False, return_loss_components=False, device=None,):
    """
    Same as other unroll function but used for reinforcement learning tasks.
    The only difference is that in reinforcement learning tasks we only have a ground truth of a single output of the network. 
    Namely, the output that corresponds to the action that was taken.
    We fix this by copying the output of the network to the missing outputs. Basically we are saying that all other outputs than the action were right.
    
    In the thesis we have not reported about RL tasks so this function is not used in the thesis.
    """
    network = inner_model

    loss_accum = None
    be_accum = 0
    be_count = 0

    log_list = []

    for i in range(K):

        (state, action), expected_value = task._get_next(policy_net=inner_states)

        if state is None:
            continue #happens when the task does not have enough data in the replay memory

        state_action_values, intermediates, be_sum = network.forward(inner_states, state)

        be_accum += be_sum
        be_count += 1

        # convert action to onehot boolean
        action = torch.nn.functional.one_hot(action, num_classes=state_action_values.shape[1]).squeeze().bool()

        # state_action_values is of shape (batch_size, num_actions)
        # expected_value is of shape (batch_size, 1)

        # duplicate state_action_values and detach it
        expected_value = state_action_values.detach() * ~action + expected_value.unsqueeze(-1).expand(expected_value.shape[0], state_action_values.shape[1]) * action # we need to do the oposite of gather
        loss = network.calculate_loss(state_action_values, expected_value)

        if loss_accum is None and i > 0:
            loss_accum = loss
        elif i > 0:
            loss_accum += loss #todo check how we need to handle the loss
        
        if return_loss_list:
            log_list.append(loss)

        weight_updates = network.backward(state_action_values, expected_value, outer_model, outer_state, inner_states, intermediates) # we also need the intermediates to do the backward pass
        inner_states = network.step(inner_states, weight_updates, outer_state) # here we get the updated inner states


    if be_count != 0:
        be_accum /= be_count #keep it between 0 and 0.5
        be_accum *= 2 #keep it between 0 and 1
        be_accum *= 1 * loss_accum.detach() #weight it to be the same as total loss
    
    if loss_accum is None:
        loss_accum = torch.tensor(0.0, device=device)
    

    to_return = [loss_accum+be_accum, inner_states]

    if return_loss_list:
        to_return.append(log_list)

    if return_final_loss:
        to_return.append(loss)

    if return_loss_components:
        to_return.append(loss_accum)
        to_return.append(be_accum)

    return tuple(to_return)
