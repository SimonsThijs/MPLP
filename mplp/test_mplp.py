import unittest

from mplp.layers import *
from mplp.network import *
from mplp.tasks import FashionMNISTTask

import torch
import torch.nn.functional as F


class TestMSELoss(unittest.TestCase):
    def test_backward_multi_output(self):
        batch_size = 3
        n_nodes_layer_before = 4 #this is the output layer that connects to the loss node

        mseloss = MessagePassingMSE()


        def g(state, x):
            return x**2


        loss_message = torch.flip(torch.arange(1,batch_size+1),dims=[0]) # a loss for each example in a batch
        loss_message = loss_message.unsqueeze(-1)
        loss_message = loss_message.repeat(1, n_nodes_layer_before)

        forward_input = torch.arange(1,batch_size*n_nodes_layer_before+1).view(batch_size, n_nodes_layer_before).unsqueeze(-1)
        mseloss.forward_input = forward_input

        y = torch.flip(torch.arange(1,batch_size*n_nodes_layer_before+1), dims=[0]).view(batch_size, n_nodes_layer_before)

        
        result = mseloss.backward(loss_message, y, g, [None,], forward_input)


        truth = torch.tensor([[[  9,   1, 144],
                            [  9,   4, 121],
                            [  9,   9, 100],
                            [  9,  16,  81]],

                            [[  4,  25,  64],
                            [  4,  36,  49],
                            [  4,  49,  36],
                            [  4,  64,  25]],

                            [[  1,  81,  16],
                            [  1, 100,   9],
                            [  1, 121,   4],
                            [  1, 144,   1]]])

        self.assertTrue(torch.equal(result, truth))
    
    def test_backward_single_output(self):
        batch_size = 3
        n_nodes_layer_before = 1 #this is the output layer that connects to the loss node

        mseloss = MessagePassingMSE()

        def g(state, x):
            return x**2

        loss_message = torch.flip(torch.arange(1,batch_size+1),dims=[0])*2 # a loss for each example in a batch
        loss_message = loss_message.unsqueeze(-1)
        loss_message = loss_message.repeat(1, n_nodes_layer_before)



        forward_input = torch.arange(1,batch_size*n_nodes_layer_before+1).view(batch_size, n_nodes_layer_before).unsqueeze(-1)


        y = torch.flip(torch.arange(1,batch_size*n_nodes_layer_before+1), dims=[0]).view(batch_size, n_nodes_layer_before)
        
        
        result = mseloss.backward(loss_message, y, g, [None,], forward_input)

        truth = torch.tensor([[[36,  1,  9]],
                            [[16,  4,  4]],
                            [[ 4,  9,  1]]])

        self.assertTrue(torch.equal(result, truth))

class TestCrossEntropyLoss(unittest.TestCase):
    def test_backward_multi_output(self):
        batch_size = 3
        n_nodes_layer_before = 4 #this is the output layer that connects to the loss node

        ce_loss = MessagePassingCrossEntropy()

        def g(state, x):
            return x**2

        loss_message = torch.flip(torch.arange(1,batch_size+1),dims=[0]) # a loss for each example in a batch
        loss_message = loss_message.unsqueeze(-1)
        loss_message = loss_message.repeat(1, n_nodes_layer_before)

        forward_input = torch.arange(1,batch_size*n_nodes_layer_before+1).view(batch_size, n_nodes_layer_before)
        forward_input = forward_input.unsqueeze(-1)


        y = torch.tensor([0,3,2])
        result = ce_loss.backward(loss_message, y, g, [None,], forward_input)

        truth = torch.tensor([[[  9,   1,   1],
                            [  9,   4,   0],
                            [  9,   9,   0],
                            [  9,  16,   0]],

                            [[  4,  25,   0],
                            [  4,  36,   0],
                            [  4,  49,   0],
                            [  4,  64,   1]],

                            [[  1,  81,   0],
                            [  1, 100,   0],
                            [  1, 121,   1],
                            [  1, 144,   0]]])

        self.assertTrue(torch.equal(result, truth))

class TestActivationLayer(unittest.TestCase):
    def test_backward(self):
        batch_size = 3
        n_nodes_out = 4 # is the same as in for an activation layer
        message_size = 3

        activation = MessagePassingActivation()

        def g(state, x):
            return torch.sqrt(x)

        receiving_messages = torch.tensor([[[  9,   1, 144], #these are the results from TestMSELoss.test_backward_multi_output
                                        [  9,   4, 121],
                                        [  9,   9, 100],
                                        [  9,  16,  81]],

                                        [[  4,  25,  64],
                                        [  4,  36,  49],
                                        [  4,  49,  36],
                                        [  4,  64,  25]],

                                        [[  1,  81,  16],
                                        [  1, 100,   9],
                                        [  1, 121,   4],
                                        [  1, 144,   1]]])
        
        # lets make sure we did not make a mistake in the test
        receiving_messages_size = receiving_messages.size()
        self.assertEqual(receiving_messages_size[0], batch_size)
        self.assertEqual(receiving_messages_size[1], n_nodes_out)
        self.assertEqual(receiving_messages_size[2], message_size)

        forward_input = torch.arange(1,batch_size*n_nodes_out+1).view((batch_size, n_nodes_out))**2
        forward_input = forward_input.unsqueeze(-1)

        result = activation.backward(receiving_messages, g, [None,], forward_input)

        truth = torch.tensor([[[ 3.,  1., 12.,  1.],
                            [ 3.,  2., 11.,  2.],
                            [ 3.,  3., 10.,  3.],
                            [ 3.,  4.,  9.,  4.]],

                            [[ 2.,  5.,  8.,  5.],
                            [ 2.,  6.,  7.,  6.],
                            [ 2.,  7.,  6.,  7.],
                            [ 2.,  8.,  5.,  8.]],

                            [[ 1.,  9.,  4.,  9.],
                            [ 1., 10.,  3., 10.],
                            [ 1., 11.,  2., 11.],
                            [ 1., 12.,  1., 12.]]])


        self.assertTrue(torch.equal(result, truth))


class TestLinearLayer(unittest.TestCase):

    def test_backward(self):
        batch_size = 3
        size_in = 2
        size_out = 4
        message_size = 3
        
        linear = MessagePassingLinear(size_in=size_in, size_out=size_out)

        def g(state, x):
            # receiving_messages, forward_input, weights_flattened

            one = x[:,0]+x[:,1]
            two = x[:,2]+x[:,3]
            three = x[:,4]

            one = torch.unsqueeze(one, dim=-1)
            two = torch.unsqueeze(two, dim=-1)
            three = torch.unsqueeze(three, dim=-1)

            result = torch.cat([one, two, three], dim=-1)
            return result


        def f(state, x):
            result = torch.sum(x, dim=-1)
            return result


        receiving_messages = torch.tensor([[[  9,   1, 144], #these are the results from TestMSELoss.test_backward_multi_output
                                [  9,   4, 121],
                                [  9,   9, 100],
                                [  9,  16,  81]],

                                [[  4,  25,  64],
                                [  4,  36,  49],
                                [  4,  49,  36],
                                [  4,  64,  25]],

                                [[  1,  81,  16],
                                [  1, 100,   9],
                                [  1, 121,   4],
                                [  1, 144,   1]]])

        weights = torch.arange(1,(size_in+1)*size_out+1, dtype=torch.float32).view((size_in+1,size_out))
    
        forward_input = torch.arange(1,batch_size*(size_in+1)+1, dtype=torch.float32).view((batch_size,size_in+1))


        result, weights = linear.backward(receiving_messages, f, [None,], g, [None,], forward_input, weights, use_all_input_features=True)



        truth = torch.tensor([[[ 66., 450.,  10.],
         [ 66., 454.,  26.]],

        [[190., 190.,  10.],
         [190., 194.,  26.]],

        [[450.,  58.,  10.],
         [450.,  62.,  26.]]])
        
        
        self.assertTrue(torch.equal(result, truth))


class TestMessagePassingNetwork(unittest.TestCase):

    def test_torch_is_equal(self):
        task = FashionMNISTTask(batch_size=16)
        x, y, _ = task.get_next_train_batch()
    
        layers = [MessagePassingLinear(size_in=784, size_out=100), MessagePassingReLU(), MessagePassingLinear(size_in=100, size_out=10)]
        loss = MessagePassingCrossEntropy()
        network = MessagePassingNetwork(layers, loss)
        inner_states = network.init_states()
        torch_network, loss = network.totorch(inner_states)
        

        output_mplp, intermediates, be = network.forward(inner_states, x)
        output_torch = torch_network(x)


        comparison =  torch.abs(output_mplp - output_torch) < 1e-6

        # check if comparison is all true
        self.assertTrue(comparison.all())






if __name__ == '__main__':
    unittest.main()




