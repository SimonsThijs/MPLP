import torch
import subprocess
import sys
from io import StringIO
import pandas as pd
import os
import numpy as np



# pytorch util to flatten a module to a vector
def flatten_module(module):
    return torch.cat([p.view(-1) for p in module.parameters()])

# pytorch util to unflatten a vector to a module
def unflatten_module(module, vec):
    idx = 0
    for p in module.parameters():
        p.data = vec[idx:idx+p.numel()].view(p.shape)
        idx += p.numel()
    
    return module


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(str(gpu_stats, 'utf-8')),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    return idx


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    elif torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        return torch.device("cuda:{}".format(free_gpu_id))
    else:
        return torch.device("cpu")



# merge dictionaroies of lists
def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)


def bfs_backwards_torch(node, apply, *args, **kwargs):
    visited = []
    queue = []
    depth = {}

    visited.append(node)
    queue.append(node)
    depth[node] = 0

    while queue:
        s = queue.pop(0)
        apply(s, depth[s], *args, **kwargs)

        for neighbour in s.next_functions:
            if neighbour[0] not in visited and neighbour[0] is not None:
                visited.append(neighbour[0])
                queue.append(neighbour[0])
                depth[neighbour[0]] = depth[s]+1


def save_results(data, path, experiment_name, run_name):
    full_path = os.path.join(path, experiment_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    filename = run_name + '.pt'

    full_path_and_filename = os.path.join(full_path, filename)
    torch.save(data, full_path_and_filename)

def experiment_has_been_run(path, experiment_name):
    full_path = os.path.join(path, experiment_name)

    return os.path.exists(full_path)


# this functions converts a list of lists of tensors to a single tensor and we keep the dimensions of the original list
def list_of_lists_to_tensor(list_of_lists):
    return torch.cat([torch.cat([tensor.unsqueeze(0) for tensor in list_of_tensors], dim=0).unsqueeze(0) for list_of_tensors in list_of_lists], dim=0)


# this helper function is used to unwrapped batched and gradient tracking tensors
def detach_numpy(tensor):
    tensor = tensor.detach().cpu()
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        tensor = torch._C._functorch.get_unwrapped(tensor)
        return np.array(tensor.storage().tolist()).reshape(tensor.shape)
    return tensor.numpy()
