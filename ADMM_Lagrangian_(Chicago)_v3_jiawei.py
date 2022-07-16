#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax.nn
import pandas as pd
import numpy as np
import sys
import jax.numpy as jnp
from scipy import optimize
from jax import jit, grad
import time
import os


# In[2]:


# Load the datasets
agent1 = pd.read_csv('trial1/agent_type_1_v2.csv')
agent2 = pd.read_csv('trial1/agent_type_2_v2.csv')
agent3 = pd.read_csv('trial1/agent_type_3_v2.csv')
agent4 = pd.read_csv('trial1/agent_type_4_v2.csv')

# Origin Layer
ozone_df = agent1
ozone_df.rename(columns={'agent_id':'ozone_id'}, inplace=True)

# Origin - Destination
od_df = agent2
od_df.rename(columns={'agent_id':'od_id'}, inplace=True)

# Path
path_df = agent3
path_df.rename(columns={'agent_id':'path_id'}, inplace=True)

# Link
link_df = agent4
link_df.rename(columns={'agent_id':'link_id'}, inplace=True)

# Reset index of each file
ozone_df.reset_index(drop=True, inplace=True)
od_df.reset_index(drop=True, inplace=True)
path_df.reset_index(drop=True, inplace=True)
link_df.reset_index(drop=True, inplace=True)

"""Set up lengths of each layer"""
num_ozone = ozone_df.shape[0]
num_od = od_df.shape[0]
num_path = path_df.shape[0]
num_link = link_df.shape[0]

"""Define dictionaries"""
# Origin layer
node_zone_dict = ozone_df[['o_node_id', 'ozone_id']].set_index('o_node_id').to_dict()['ozone_id']

# Origin-Destination layer
od_df['od_pair'] = od_df.apply(lambda x: (int(x.o_zone_id), int(x.d_zone_id)), axis=1)
od_df['ozone_id'] = od_df.apply(lambda x: node_zone_dict[int(x.o_zone_id)], axis=1)
od_pair_dict = od_df[['od_pair', 'od_id']].set_index('od_pair').to_dict()['od_id']

# Path to Origin-Destination
path_df['od_id'] = path_df.apply(lambda x: od_pair_dict[int(x.o_zone_id), int(x.d_zone_id)], axis=1)
path_od_dict = path_df[['path_id', 'od_id']].set_index('path_id').to_dict()['od_id']

# Link layer
link_df['link_pair'] = link_df.apply(lambda x: (int(x.from_node_id), int(x.to_node_id)), axis=1)  
link_id_pair_dict = link_df[['link_id', 'link_pair']].set_index('link_pair').to_dict()['link_id']

# Origin-Destination to Origin
od_ozone_dict = od_df[['ozone_id', 'od_id']].set_index('od_id').to_dict()['ozone_id']

# 1. Generate the orgin-destination volume
od_volume = jnp.array(agent2['od_demand'])
od_volume = od_volume.reshape(od_volume.shape[0], 1)

# 2. Generate the od-path incidence matrix
# To count the number of x_f flow variables
count_x_f = []
for i in path_df['o_node_id'].unique():
    
    for j in path_df['d_node_id'].unique():
        
        index = path_df[(path_df['o_node_id'] == i) & (path_df['d_node_id'] == j)].index
        
        if len(index)>0:
            
            for k in range(len(index)):
                
                if k==len(index)-1: 
                    
                    count_x_f.append(k)

# the size of the column
num_val = np.array(count_x_f).sum()
od_path_inc = np.zeros([od_volume.shape[0], num_val], dtype=np.float64)
val_loc = np.where(np.array(count_x_f)>0)

start_index = 0
for i in val_loc[0]:

  od_path_inc[i][start_index: start_index+ count_x_f[i]] = 1

  start_index = start_index + count_x_f[i]

# 3. Generate the path-link incidence matrix
path_link_inc_mat = np.zeros([num_path, num_link], dtype=np.float64)
for i in range(num_path):
    path_r = path_df.loc[i]
    node_list = path_r.node_sequence.split(';')
    for link_l in range(len(node_list) - 1):
        link_pair = (int(node_list[len(node_list) - 1 - link_l]), int(node_list[len(node_list) - 2 - link_l]))
        link_id = link_id_pair_dict[link_pair]
        path_link_inc_mat[int(path_r.path_id - 1)][int(link_id - 1)] = 1.0

# To find the path flow index 
path_link_index = []
path_link_index2 = []
for i in path_df['o_node_id'].unique():
    
    for j in  np.sort(path_df['d_node_id'].unique()):
        
        index = np.array(path_df[(path_df['o_node_id'] == i) & (path_df['d_node_id'] == j)].index)

        if len(index)==1:
          path_link_index2.append(index)

        if len(index)>1:
          n_1_index = index[0 : -1]
          path_link_index.append(n_1_index)
          # print(index[-1:])

          path_link_index2.append(index[-1:])

index_1 = np.concatenate(path_link_index)
index_2 = np.concatenate(path_link_index2)

path_link_inc = path_link_inc_mat[index_1]
path_link_inc_n = path_link_inc_mat[index_2]

# Transit Costs
transit_cost = jnp.array(od_df['od_time'] * 1.2)
transit_cost = transit_cost.reshape(transit_cost.shape[0], 1)
# Link Features (free flow travel time (fftt) and capacity (cap))
fftt = jnp.array(link_df['VDF_fftt1'])
fftt = fftt.reshape(fftt.shape[0], 1)
cap  = jnp.array(link_df['VDF_cap1'])
cap = cap.reshape(cap.shape[0], 1)


# In[3]:


od_path_inc_for_min = (jnp.ones(jnp.shape(od_path_inc)) - od_path_inc) * 1e5 + od_path_inc

def calculateCoreVars(access_cost, path_flow):
    access_cost = jnp.reshape(access_cost,(-1,1))
    path_flow = jnp.reshape(path_flow, (-1,1))

    auto_volume = od_volume * jnp.exp(-theta*access_cost) / (jnp.exp(-theta*access_cost) + jnp.exp(-theta*transit_cost))

    path_flow_n = auto_volume - jnp.matmul(od_path_inc, path_flow)

    link_flow = jnp.matmul(path_link_inc.T, path_flow) + jnp.matmul(path_link_inc_n.T, path_flow_n)

    link_cost = fftt*(1 + bpr_alpha*(link_flow/cap)**bpr_beta)

    path_cost = jnp.matmul(path_link_inc, link_cost)
    path_cost_n = jnp.matmul(path_link_inc_n, link_cost)

    ue_cost0 = jnp.reshape(jnp.min(jnp.multiply(od_path_inc_for_min, jnp.reshape(path_cost, (-1,))), axis=1), (-1,1))
    ue_cost = jnp.reshape(jnp.min(jnp.concatenate([ue_cost0, path_cost_n], axis=1), axis=1), (-1,1))

    path_flow_all = jnp.concatenate([path_flow, path_flow_n], axis=0)

    flow_condi = auto_volume - path_flow_n

    cost_condi = access_cost - ue_cost

    return path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all


def calculateLoss(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve):
    path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost, path_flow)

    loss = jnp.sum(path_flow * (path_cost - jnp.matmul(od_path_inc.T, ue_cost))) + jnp.sum(path_flow_n.T * (path_cost_n - ue_cost))\
         + jnp.sum(jnp.multiply(lambda_cost, cost_condi)) + jnp.sum((rho_/2)*cost_condi**2)\
         + jnp.sum(jnp.multiply(lambda_flow_bal, flow_condi)) + jnp.sum((rho_/2)*flow_condi**2)\
         + jnp.sum(jnp.multiply(lambda_positve, jax.nn.relu(-path_flow_all))) + jnp.sum((rho_/2)*jax.nn.relu(-path_flow_all)**2)

    return loss


# In[4]:
@jit
def calculateLossAccessCost(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve):
    return calculateLoss(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve)

@jit
def calculateLossPathFlow(path_flow, access_cost, lambda_cost, lambda_flow_bal, lambda_positve):
    return calculateLoss(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve)

@jit
def lambda_condi(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve):
    _, _, _, _, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost, path_flow)

    lambda_cost = lambda_cost + rho_ * cost_condi
    lambda_flow_bal = lambda_flow_bal + rho_ * flow_condi
    lambda_positve = lambda_positve + rho_ * rho_ * (jax.nn.relu(-path_flow_all))

    return lambda_cost, lambda_flow_bal, lambda_positve


# In[6]:


t1 = time.perf_counter()
# Set up the optimization formulation
theta = 0.1
bpr_alpha = 0.15
bpr_beta = 4
lambda_cost = jnp.zeros(len(od_path_inc),) #6
lambda_flow_bal = jnp.zeros(len(od_path_inc),) #6
lambda_positve = jnp.zeros(len(path_link_inc_n) + len(path_link_inc)) #9

# Initialize the Lagrangian multipliers
lambda_cost = lambda_cost.reshape(lambda_cost.shape[0], 1)
lambda_flow_bal = lambda_flow_bal.reshape(lambda_flow_bal.shape[0], 1)
lambda_positve = lambda_positve.reshape(lambda_positve.shape[0], 1)

rho_ = 1.0

# Initialize the variables
access_cost = jnp.array(np.random.randint(low=1, high=10, size=(len(od_path_inc),)))
path_flow =   jnp.array(np.random.randint(low=1, high=100, size=(len(path_link_inc),)))

for i in range(50):
    # Demand Component
    demand_opti = optimize.minimize(calculateLossAccessCost,
                            access_cost,
                            jac=jit(grad(calculateLossAccessCost)),
                            args=(path_flow, lambda_cost, lambda_flow_bal, lambda_positve),
                            method='BFGS')
    updated_cost = demand_opti.x
    
    # Supply Component
    supply_opti = optimize.minimize(calculateLossPathFlow,
                            path_flow,
                            jac=jit(grad(calculateLossPathFlow)),
                            args=(updated_cost, lambda_cost, lambda_flow_bal, lambda_positve),
                            method='BFGS')

    updated_path = supply_opti.x

    updated_lambdas = lambda_condi(updated_cost, updated_path, 
                                  lambda_cost, 
                                  lambda_flow_bal, 
                                  lambda_positve)
    
    # Update the states
    access_cost = updated_cost
    path_flow   = updated_path
    lambda_cost = updated_lambdas[0]
    lambda_flow_bal = updated_lambdas[1]
    lambda_positve = updated_lambdas[2]
    
    print('iteration_step:', i)
    print('1. Updating the cost consistency:', access_cost)
    print('2. Updating the optimal path flows:', path_flow)
    print()
    
    """TODO:
    1. Set up a large step number to find the point to stop the iterative process
    """

t2 = time.perf_counter()
print('Running Time:', t2-t1)


# In[7]:


def AMPL_sol_check(access_cost, path_flow):

    access_cost = jnp.reshape(access_cost,(-1,1))
    path_flow = jnp.reshape(path_flow, (-1,1))

    auto_volume = od_volume * jnp.exp(-theta*access_cost) / (jnp.exp(-theta*access_cost) + jnp.exp(-theta*transit_cost))

    path_flow_n = auto_volume - jnp.matmul(od_path_inc, path_flow)

    link_flow = jnp.matmul(path_link_inc.T, path_flow) + jnp.matmul(path_link_inc_n.T, path_flow_n)

    link_cost = fftt*(1 + bpr_alpha*(link_flow/cap)**bpr_beta)

    path_cost = jnp.matmul(path_link_inc, link_cost)
    path_cost_n = jnp.matmul(path_link_inc_n, link_cost)

    ue_cost0 = jnp.reshape(jnp.min(jnp.multiply(od_path_inc_for_min, jnp.reshape(path_cost, (-1,))), axis=1), (-1,1))
    ue_cost = jnp.reshape(jnp.min(jnp.concatenate([ue_cost0, path_cost_n], axis=1), axis=1), (-1,1))

    # Need to add the constraints
    # print('access costs:', access_cost)
    # print()
    # print('path flows:', jnp.concatenate([path_flow, path_flow_n], axis=0))

    path_flow_all = jnp.concatenate([path_flow, path_flow_n], axis=0)

    
    return path_flow_all, path_cost, path_cost_n
# path_flow_result = pd.DataFrame(AMPL_sol_check(access_cost, path_flow)[0])
# path_flow_result.rename(columns={0:'path_flow'})
# access_cost_result = pd.DataFrame(AMPL_sol_check(access_cost, path_flow)[1])
# access_cost_result.rename(columns={0:'path cost'})


# In[8]:


pd.DataFrame(access_cost)

