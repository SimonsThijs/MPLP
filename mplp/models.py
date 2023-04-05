import torch
import torch.nn as nn


"""
The most basic models for the MPLP. These models are not always used, sometimes we define custom models.
"""

class LearningRate(nn.Module):
    def __init__(self, lr, device='cpu'):
        super(LearningRate, self).__init__()
        self.lr = torch.nn.Parameter(torch.tensor(lr))
        self.to(device)

class UpdateFunction(nn.Module):
    def __init__(self, message_size, hidden_size, output_size=1, device='cpu'):
        super(UpdateFunction, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(message_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x)

class MessageLinear(nn.Module):
    def __init__(self, message_size, hidden_size, device='cpu'):
        super(MessageLinear, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(message_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x) 


class MessageSoftmax(nn.Module):
    def __init__(self, message_size, hidden_size, device='cpu'):
        super(MessageSoftmax, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(message_size+1, hidden_size),
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
            nn.Linear(message_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x) 

class MessageTanh(nn.Module):
    def __init__(self, message_size, hidden_size, device='cpu'):
        super(MessageTanh, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(message_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x) 

class MessageLoss(nn.Module):
    def __init__(self, message_size, hidden_size, input_size=3, device='cpu'):
        super(MessageLoss, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            )
        
        self.to(device)

    def forward(self, x):
        return self.func(x) 







"""
These models are used to mimic SGD, they are explained in my thesis in chapter: 'Expressiveness to imitate SGD'
"""


class NonParameterizedModel():
    """
    Base class used to mimic SGD
    """
    def __init__(self, device='cpu'):
        self.device = device
    
    @staticmethod
    def func(s,b, input):
        raise NotImplementedError

class SGDUpdateFunction(NonParameterizedModel):

    @staticmethod
    def func(s,b, input):
        m = input[:,0]
        x = input[:,1]

        result = x*m
        return torch.unsqueeze(result, dim=-1)

class SGDMessageLinear(NonParameterizedModel):

    @staticmethod
    def func(s,b, input):
        m = input[:,0]
        w = input[:,1]

        result = w*m
        return torch.unsqueeze(result, dim=-1)

class SGDMessageReLUActivation(NonParameterizedModel):

    @staticmethod
    def func(s,b, input):
        def relu_derivative(x):
            return (x > 0).float()

        m = input[:,0]
        x = input[:,1]

        x = relu_derivative(x)
        result = x*m
        return torch.unsqueeze(result, dim=-1)

class SGDMessageSoftmaxActivation(NonParameterizedModel):

    @staticmethod
    def func(s,b, input):

        def softmax_derivative(x):
            return x*(1-x)

        m = input[:,0]
        x = input[:,1]

        x = softmax_derivative(x)
        result = x*m
        return torch.unsqueeze(result, dim=-1)

class SGDMessageMSELoss(NonParameterizedModel):

    @staticmethod
    def func(s,b, input):
        m = input[:,0]
        guess = input[:,1]
        y = input[:,2]

        result = guess-y
        return torch.unsqueeze(result, dim=-1)   

class SGDMessageCrossentropyLoss(NonParameterizedModel):
    """Appendix C in the thesis"""

    @staticmethod
    def func(s,b, input):
        m = input[:,0]
        probs = input[:,1]
        y = input[:,2]

        result = probs-y
        return torch.unsqueeze(result, dim=-1)    


    

