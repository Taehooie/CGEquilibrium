import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad, jit

from jax import random
key = random.PRNGKey(10)

import scipy
import numpy as np

# Variables to be optimized (initial settings)
c_d = 20.0
x_path1 =  79.33806311
x_path2 =  125.1719369

# Initialize Lagrange multipliers
lambda_1 = 0.0
lambda_2 = 0.0
lambda_list = [lambda_1, lambda_2]

@jit
def obj_func(access_cost, x_paths, lambda_list):

    '''
    Return
        Compute the loss function of the augmented lagrangian relaxation
        
    Parameters
    ----------
    access_cost : numpy.array
        The updated accessibility cost (nx1)
    x_paths : numpy.array
        The updated path flows of the two-link network (nx2)
    lambda_list : numpy.array
        The updated lagrangian multipliers (nx2)
    
    '''
    tot_origin_flow = 300.0 # assumption: travelers are selecting the auto-mode
    theta = 1.0
    transit_cost = 25.0

    auto_probs = (jnp.exp(-theta*access_cost[0])/(jnp.exp(-theta*access_cost[0])+jnp.exp(-theta*transit_cost)))

    # Link function parameters
    ttff_1, cap_1 = 20, 4500
    ttff_2, cap_2 = 30, 3000

    # Link travel time function (BPR)
    t_1 = ttff_1*(1 + 0.15*(x_paths[0]/cap_1)**4)
    t_2 = ttff_2*(1 + 0.15*(x_paths[1]/cap_2)**4)

    # Path flow travel time
    t_p1 = t_1
    t_p2 = t_2

    # compute min of travel time
    c_s =  jnp.min(jnp.array([t_p1, t_p2]))

    # Volume
    tot_vol = tot_origin_flow*auto_probs
    path_x = jnp.stack([x_paths[0], x_paths[1]], 0) #prediction

    # # Time
    cost_demand = access_cost[0]
    path_t = jnp.stack([t_p1, t_p2], 0) 

    # Constraints to penalize the costs
    link_condi = access_cost[0] - c_s # integrate demand and supply (c_d - c_s)
    flow_condi = tot_vol - x_paths[0] -  x_paths[1] # flow constraint

    ested_val = [path_x, 
                 path_t, 
                 c_s, 
                 link_condi, 
                 flow_condi]
    
    rho_ = 0.001
    # 0.01 indicates the arbitrary rho value
    loss = path_x[0]*(path_t[0] - c_s) + path_x[1]*(path_t[0] - c_s) + \
           lambda_list[0]*link_condi + rho_*link_condi**2 + lambda_list[1]*flow_condi + rho_*flow_condi**2

    return loss 

@jit
def lambda_condi(up_access_cost, up_x_paths, lambda_list):
    '''
    Return
        Update the lagrangian multipliers corresponding to the accessibility cost and the optimal path flows
        
    Parameters
    ----------
    up_access_cost : numpy.array
        The updated accessibility cost (nx1)
    up_x_paths : numpy.array
        The updated path flows of the two-link network (nx2)
    lambda_list : numpy.array
        The updated lagrangian multipliers (nx2)

    '''
    tot_origin_flow = 300.0 
    theta = 1.0
    transit_cost = 25.0

    auto_probs = (jnp.exp(-theta*up_access_cost[0])/(jnp.exp(-theta*up_access_cost[0])+jnp.exp(-theta*transit_cost)))

    # Link function parameters
    ttff_1, cap_1 = 20, 4500
    ttff_2, cap_2 = 30, 3000

    # Link travel time function (BPR)
    t_1 = ttff_1*(1 + 0.15*(up_x_paths[0]/cap_1)**4)
    t_2 = ttff_2*(1 + 0.15*(up_x_paths[1]/cap_2)**4)

    # Path flow travel time
    t_p1 = t_1
    t_p2 = t_2

    # compute min of travel time
    c_s =  jnp.min(jnp.array([t_p1, t_p2]))
    
    # Volume
    tot_vol = tot_origin_flow*auto_probs

    # # Time
    cost_demand = up_access_cost[0]
    path_t = jnp.stack([t_p1, t_p2], 0) 
    print(cost_demand)
    # Constraints to penalize the costs
    link_condi = up_access_cost[0] - c_s # integrate demand and supply (c_d - c_s)
    flow_condi = tot_vol - up_x_paths[0] -  up_x_paths[1] # flow constraint
    
    ested_val = [link_condi, flow_condi]
    
    rho_ = 0.001
    lambda_list[0] = lambda_list[0] + rho_ * link_condi
    lambda_list[1] = lambda_list[1] + rho_ * flow_condi
    
    return lambda_list

"""TO DO: Formulate the convergence check function and replace 'for' loop to 'while' """
# Initialize the accessibility cost and the path flows
access_cost = jnp.array([c_d])
x_paths  = jnp.array([x_path1, x_path2])
for i in range(200):
    '''
    Loop over the augmented lagrangian relaxation using the ADMM algorithm
    
    1. Update the accessibility cost
    2. Update the path flows 
    3. Update the lagrangian multipliers
    
    '''
    # Optimization to train the demand component # compute c_d(k+1) fixing x(k)
    train_demand = scipy.optimize.minimize(obj_func, 
                                             access_cost, 
                                             jac  = jit(jax.grad(obj_func)),
                                             args = (x_paths, lambda_list),
                                             method='BFGS')
    updated_cost = jnp.array(train_demand.x)

    # set up the shape of the variables
    access_cost = jnp.array(train_demand.x) # compute a list of x(k+1) by fixing c_d(k+1)
    x_paths  = jnp.array([x_path1, x_path2])
    # Optimization to train the supply component
    train_supply = scipy.optimize.minimize(obj_func, 
                                             x_paths, 
                                             jac  = jit(jax.grad(obj_func)),
                                             args = (updated_cost, lambda_list),
                                             method='BFGS')

    updated_paths = train_supply.x
    # After this step, we update c_d(k) -> c_d(k+1) and x_s(k) -> x_s(k+1).
    # Then the next step is updating the lambda value (\lambda(k) -> \lambda(k+1))
    # lambda(k+1) = lambda(k) + rho*(defined conditions(e.g., link_condi or flow_condi))
    updated_lambdas = lambda_condi(updated_cost, updated_paths, lambda_list)
    
    # Reassign the updated parameters and variables including the lagrangian multipliers
    lambda_list = updated_lambdas
    access_cost = updated_cost
    x_paths = updated_paths
    
    # Print out the sequentially updated results
    print('iteration_step:', i)
    print('1. Updating the generalized travel costs:', updated_cost)
    print('2. Updating the optimal path flows:', updated_paths)
    print('3. Updating the lambda multiplier for the cost linkage:', lambda_list[0])
    print('4. Updating the lambda multiplier for the flow balance:', lambda_list[1])
    print()