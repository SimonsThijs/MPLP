"""
This file also contains some very experimental code for calculating the gradients using evolution strategies.
"""

import functorch
import torch

from mplp.util import *

def analytical_gradient(unroll_fn, K, inner_states, outer_states, inner_model, outer_model, task):
    """
    Calculates the gradients in the analytical way. This is basically what (T)BPTT uses. 
    We only use this method for calculating gradients currently.
    """
    grad_and_value_unroll_fn = functorch.grad_and_value(unroll_fn, argnums=1, has_aux=True)
    grads, (value, new_inner_states) = grad_and_value_unroll_fn(inner_states, outer_states, K, inner_model, outer_model, task)
    return grads, new_inner_states










def pes_grad(unroll_fn_vmapped, K, N, inner_states, outer_states, inner_model, outer_model, task, sigma, pert_accums, device='cpu'):
    """
    This function calculates the gradients of the outer loss with respect to the outer parameters. 
    We need to keep track of the pertubations somewhere. 
    """

    def calc_perturbations(tensor):
        perts = torch.randn((N // 2, *tensor.shape[1:]), device=device) * sigma
        perts = torch.cat((perts, -perts), dim=0)
        return perts

    pertubations = apply_to_nested(calc_perturbations, outer_states)
    perturbed_outer_states = apply_to_two_nested(lambda x, y: y+x, outer_states, pertubations)


    with torch.no_grad():
        losses, new_inner_states = unroll_fn_vmapped(inner_states, perturbed_outer_states, K, inner_model, outer_model, task, device=device)

    if pert_accums is None:
        pert_accums = pertubations
    else:
        pert_accums = apply_to_two_nested(lambda x, y: x + y, pert_accums, pertubations)

    N_times_sigma_squared = N * (sigma ** 2)
    def calc_grad(pert_accum):
        shape = pert_accum.shape
        loss_reshaped_for_broadcast = losses.reshape(N, *((1,)*(len(shape)-1)))
        result = torch.sum(pert_accum * loss_reshaped_for_broadcast, dim=0) / N_times_sigma_squared
        return result

    grad_estimate = apply_to_nested(calc_grad, pert_accums)

    return grad_estimate, new_inner_states, pert_accums


def es_grad(unroll_fn_vmapped, K, N, inner_states, outer_states, inner_model, outer_model, task, sigma, device='cpu'):
    """
    This function calculates the gradients of the outer loss with respect to the outer parameters. 
    """

    def calc_perturbations(tensor):
        perts = torch.randn((N // 2, *tensor.shape), device=device) * sigma
        perts = torch.cat((perts, -perts), dim=0)
        return perts


    def add(a,b):
        result = a.unsqueeze(0)+b
        return result

    pertubations = apply_to_nested(calc_perturbations, outer_states)
    perturbed_outer_states = apply_to_two_nested(add, outer_states, pertubations)

    with torch.no_grad():
        losses, new_inner_states = unroll_fn_vmapped(inner_states, perturbed_outer_states, K, inner_model, outer_model, task, device=device)

    N_times_sigma_squared = N * (sigma ** 2)
    def calc_grad(pertubations):
        shape = pertubations.shape
        loss_reshaped_for_broadcast = losses.reshape(N, *((1,)*(len(shape)-1)))
        result = torch.sum(pertubations * loss_reshaped_for_broadcast, dim=0) / N_times_sigma_squared
        return result

    grad_estimate = apply_to_nested(calc_grad, pertubations)

    return grad_estimate, new_inner_states


def analytical_smoothed_gradient(unroll_fn, K, N, inner_states, outer_states, inner_model, outer_model, task, sigma, device='cpu'):

    def calc_perturbations(tensor):
        perts = torch.randn((N // 2, *tensor.shape[1:]), device=device) * sigma
        perts = torch.cat((perts, -perts), dim=0)
        return perts

    pertubations = apply_to_nested(calc_perturbations, outer_states)
    perturbed_outer_states = apply_to_two_nested(lambda x, y: x + y, outer_states, pertubations)
    
    grad_and_value_unroll_fn = functorch.grad_and_value(unroll_fn, argnums=1, has_aux=True)
    grad_and_value_unroll_fn_vmapped = functorch.vmap(grad_and_value_unroll_fn, in_dims=(0, 0, None, None, None, None), out_dims=(0, 0), randomness='same')



    grads, (value, new_inner_states) = grad_and_value_unroll_fn_vmapped(inner_states, perturbed_outer_states, K, inner_model, outer_model, task, device)

    def calc_grad(grads):
        return torch.sum(grads, dim=0)/N
    grad_average = apply_to_nested(calc_grad, grads)

    return grad_average, new_inner_states

