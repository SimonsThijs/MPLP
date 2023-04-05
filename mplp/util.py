import torch
import torch.nn as nn
import functorch
import numpy as np

from mplp.models import NonParameterizedModel

def object_is_activation(obj):
    list_of_activation_classes = [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax]
    for activation_class in list_of_activation_classes:
        if isinstance(obj, activation_class):
            return True

def nonparameterizedmodel_to_state_func_tuple(model):
    func = model.func
    params = ()
    buffers = ()
    return func, params, buffers

def repeat_from_dict_and_combine_state_for_ensemble(dict, N):
    funcs = {}
    states = {}
    for key, value in dict.items():
        if isinstance(value, NonParameterizedModel):
            func, params, buffer = nonparameterizedmodel_to_state_func_tuple(value) # does nothing with N because it has no parameters
        else:
            func, params, buffer = functorch.combine_state_for_ensemble([value for _ in range(N)])

        funcs[key] = func
        states[key] = (params, buffer)

    return funcs, states
  
def list_of_dicts_combine_state_for_ensemble(models):
    funcs = {}
    states = {}
    for key, value in models[0].items():
        if isinstance(value, NonParameterizedModel):
            func, params, buffer = nonparameterizedmodel_to_state_func_tuple(value) # does nothing with N because it has no parameters
        else:
            func, params, buffer = functorch.combine_state_for_ensemble([m[key] for m in models])

        funcs[key] = func
        states[key] = (params, buffer)

    return funcs, states

def apply_to_nested(func, data):
  """
  Applies a function to the elements of a nested structure, recursively.

  Args:
    func: The function to apply to the elements of the structure.
    data: The nested structure. Can be a list, tuple, set, or dictionary.

  Returns:
    The modified nested structure, with the function applied to all elements.
  """

  # Check the type of the input data
  if isinstance(data, (list, tuple, set)):
    # If the data is a list, tuple, or set, apply the function to each element
    # and recursively call the function on any sub-structures
    return type(data)(apply_to_nested(func, elem) for elem in data)
  elif isinstance(data, dict):
    # If the data is a dictionary, apply the function to the values and recursively
    # call the function on any sub-structures
    return {key: apply_to_nested(func, value) for key, value in data.items()}
  else:
    # If the data is not a collection, apply the function to the data and return it
    return func(data)


def apply_to_two_nested(func, data1, data2):
  """
  Applies a function to the corresponding elements of two nested structures, recursively.

  Args:
    func: The function to apply to the elements of the structures.
    data1: The first nested structure. Can be a list, tuple, set, or dictionary.
    data2: The second nested structure. Must have the same structure as data1.

  Returns:
    The modified nested structures, with the function applied to the corresponding elements.
  """

  # Check the type of the input data
  if isinstance(data1, (list, tuple, set)) and isinstance(data2, (list, tuple, set)):
    # If the data is a list, tuple, or set, apply the function to the corresponding elements
    # and recursively call the function on any sub-structures
    return type(data1)(apply_to_two_nested(func, elem1, elem2) for elem1, elem2 in zip(data1, data2))
  elif isinstance(data1, dict) and isinstance(data2, dict):
    # If the data is a dictionary, apply the function to the corresponding values and recursively
    # call the function on any sub-structures
    return {key: apply_to_two_nested(func, value1, value2) for key, value1, value2 in zip(data1.keys(), data1.values(), data2.values())}
  else:
    # If the data is not a collection, apply the function to the data and return it
    return func(data1, data2)


def apply_to_three_nested(func, data1, data2, data3):
    """
    Applies a function to the corresponding elements of three nested structures, recursively.
    
    Args:
        func: The function to apply to the elements of the structures.
        data1: The first nested structure. Can be a list, tuple, set, or dictionary.
        data2: The second nested structure. Must have the same structure as data1.
        data3: The third nested structure. Must have the same structure as data1.
    
    Returns:
        The modified nested structures, with the function applied to the corresponding elements.
    """
    
    # Check the type of the input data
    if isinstance(data1, (list, tuple, set)) and isinstance(data2, (list, tuple, set)) and isinstance(data3, (list, tuple, set)):
        # If the data is a list, tuple, or set, apply the function to the corresponding elements
        # and recursively call the function on any sub-structures
        return type(data1)(apply_to_three_nested(func, elem1, elem2, elem3) for elem1, elem2, elem3 in zip(data1, data2, data3))
    elif isinstance(data1, dict) and isinstance(data2, dict) and isinstance(data3, dict):
        # If the data is a dictionary, apply the function to the corresponding values and recursively
        # call the function on any sub-structures
        return {key: apply_to_three_nested(func, value1, value2, value3) for key, value1, value2, value3 in zip(data1.keys(), data1.values(), data2.values(), data3.values())}
    else:
        # If the data is not a collection, apply the function to the data and return it
        return func(data1, data2, data3)


def torch_timing(string, device, f, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    ret = f(*args, **kwargs)
    torch.cuda.synchronize(device)
    end.record()
    torch.cuda.synchronize()
    # print function name and the time elapsed

    name = f.__name__ if hasattr(f, '__name__') else 'unknown'

    print(string, name, start.elapsed_time(end))
    return ret


# this helper function is used to unwrapped batched and gradient tracking tensors
def detach_numpy(tensor):
    tensor = tensor.detach().cpu()
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        tensor = torch._C._functorch.get_unwrapped(tensor)
        return np.array(tensor.storage().tolist()).reshape(tensor.shape)
    return tensor.numpy()
