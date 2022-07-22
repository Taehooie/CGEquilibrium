import pandas as pd
import numpy as np
import sys
import jax.numpy as jnp
from scipy import optimize
import time
import os
import tensorflow as tf
import tensorflow_probability as tfp

# Load the datasets
agent1 = pd.read_csv('agent_type_1_v2.csv')
agent2 = pd.read_csv('agent_type_2_v2.csv')
agent3 = pd.read_csv('agent_type_3_v2.csv')
agent4 = pd.read_csv('agent_type_4_v2.csv')

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
transit_cost = np.array(od_df['od_time'] * 1.2)
transit_cost = transit_cost.reshape(transit_cost.shape[0], 1)
# Link Features (free flow travel time (fftt) and capacity (cap))
fftt = np.array(link_df['VDF_fftt1'])
fftt = fftt.reshape(fftt.shape[0], 1)
cap  = np.array(link_df['VDF_cap1'])
cap = cap.reshape(cap.shape[0], 1)

# Initialize the variables
# Set up the optimization formulation
theta = 0.1
bpr_alpha = 0.15
bpr_beta = 4
lambda_cost = tf.zeros(len(od_path_inc))  #6
lambda_flow_bal = tf.zeros(len(od_path_inc)) #6
lambda_positve = tf.zeros(len(path_link_inc_n) + len(path_link_inc)) #9

# Initialize the Lagrangian multipliers
lambda_cost = tf.cast(tf.reshape(lambda_cost, (lambda_cost.shape[0], 1)), tf.float32)
lambda_flow_bal = tf.cast(tf.reshape(lambda_flow_bal, (lambda_flow_bal.shape[0], 1)), tf.float32)
lambda_positve = tf.cast(tf.reshape(lambda_positve, (lambda_positve.shape[0], 1)), tf.float32)

rho_ = 1.0

# Initialize the variables
access_cost = []
path_flow   = []
# Gen tf.Variable
for i in range(len(od_path_inc)):
  gen_val = tf.Variable(np.random.randint(1, 10), dtype=tf.float32)
  access_cost.append(gen_val)

for j in range(len(path_link_inc)):
  gen_val = tf.Variable(np.random.randint(1, 100), dtype=tf.float32)
  path_flow.append(gen_val)

od_path_inc_for_min = (tf.ones(tf.shape(od_path_inc)) - od_path_inc) * 1e5 + od_path_inc
@tf.function
def calculateCoreVars(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc):

    access_cost = tf.reshape(access_cost,(-1,1))
    path_flow = tf.reshape(path_flow, (-1,1))

    auto_volume = od_volume * (tf.exp(-theta*access_cost) + tf.cast(tf.exp(-theta*transit_cost), tf.float32))

    path_flow_n = auto_volume - tf.matmul(tf.cast(od_path_inc,  tf.float32), path_flow)

    path_link_inc = tf.cast(path_link_inc, tf.float32)
    path_link_inc_n = tf.cast(path_link_inc_n, tf.float32)

    link_flow = tf.matmul(tf.transpose(path_link_inc), path_flow) + tf.matmul(tf.transpose(path_link_inc_n), path_flow_n)

    link_cost = fftt*(1 + bpr_alpha*(link_flow/cap)**bpr_beta)

    path_cost = tf.matmul(path_link_inc, link_cost)
    path_cost_n = tf.matmul(path_link_inc_n, link_cost)

    ue_cost0 = tf.reshape(tf.reduce_min(tf.multiply(od_path_inc_for_min, tf.reshape(path_cost, (-1,))), axis=1), (-1,1))
    ue_cost = tf.reshape(tf.reduce_min(tf.concat([ue_cost0, path_cost_n], axis=1), axis=1), (-1,1))

    path_flow_all = tf.concat([path_flow, path_flow_n], axis=0)

    flow_condi = auto_volume - path_flow_n

    cost_condi = access_cost - ue_cost

    od_path_inc = tf.cast(od_path_inc, tf.float32)

    return path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all

@tf.function
def calculateLoss(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, 
                  lambda_cost, lambda_flow_bal, lambda_positve):

    path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc)

    od_path_inc = tf.cast(od_path_inc, tf.float32)

    loss = tf.reduce_sum(path_flow * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) + tf.reduce_sum(tf.transpose(path_flow_n) * (path_cost_n - ue_cost))\
         + tf.reduce_sum(tf.multiply(lambda_cost, cost_condi)) + tf.reduce_sum((rho_/2)*cost_condi**2)\
         + tf.reduce_sum(tf.multiply(lambda_flow_bal, flow_condi)) + tf.reduce_sum((rho_/2)*flow_condi**2)\
         + tf.reduce_sum(tf.multiply(lambda_positve, tf.nn.relu(-path_flow_all))) + tf.reduce_sum((rho_/2)*tf.nn.relu(-path_flow_all)**2)

    return loss

@tf.function
def lambda_condi(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve, path_link_inc, path_link_inc_n, od_path_inc):
    _, _, _, _, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc)

    lambda_cost = lambda_cost + rho_ * cost_condi
    lambda_flow_bal = lambda_flow_bal + rho_ * flow_condi
    lambda_positve = lambda_positve + rho_ * rho_ * (tf.nn.relu(-path_flow_all))

    return lambda_cost, lambda_flow_bal, lambda_positve

def loss_gradient_demand(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, 
                         lambda_cost, lambda_flow_bal, lambda_positve):   
      
    shapes = tf.shape_n(access_cost)

    n_tensors = len(shapes)
    count = 0
    idx = []  # stitch indices
    part = [] # partition indices
    
    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n 
    
    @tf.function
    def assign_new_model_parameters(params_1d):

        pparams = tf.dynamic_partition(params_1d, part, n_tensors)

        for i, (shape, param) in enumerate(zip(shapes, pparams)):
            access_cost[i].assign(tf.reshape(param, shape))                  
    
    @tf.function
    def est_grad(params_1d):
        # Derive the Tensorflow gradient
        with tf.GradientTape() as tape: 
            
            # Call the function to update and convert the shape of parameters
            assign_new_model_parameters(params_1d)

            # Call the loss function
            loss_value = calculateLoss(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, 
                                       lambda_cost, lambda_flow_bal, lambda_positve)
            
        # Calculate the gradient for each parameter

        estimated_grad = tape.gradient(loss_value, access_cost)
        grads_1dim = tf.dynamic_stitch(idx, estimated_grad)
        return loss_value, grads_1dim
    est_grad.idx = idx
    return est_grad

def loss_gradient_supply(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, 
                         lambda_cost, lambda_flow_bal, lambda_positve):   

    
    shapes = tf.shape_n(path_flow)
    n_tensors = len(shapes)
    count = 0
    idx = []  # stitch indices
    part = [] # partition indices
    
    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n 
    
    @tf.function
    def assign_new_model_parameters(params_1d):

        pparams = tf.dynamic_partition(params_1d, part, n_tensors)

        for i, (shape, param) in enumerate(zip(shapes, pparams)):
            path_flow[i].assign(tf.reshape(param, shape))                  
    
    @tf.function
    def est_grad(params_1d):
        # Derive the Tensorflow gradient
        with tf.GradientTape() as tape: 
            
            # Call the function to update and convert the shape of parameters
            assign_new_model_parameters(params_1d)

            # Call the loss function
            loss_value = calculateLoss(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, 
                                       lambda_cost, lambda_flow_bal, lambda_positve)
            
        # Calculate the gradient for each parameter

        estimated_grad = tape.gradient(loss_value, path_flow)
        grads_1dim = tf.dynamic_stitch(idx, estimated_grad)
        return loss_value, grads_1dim
    est_grad.idx = idx
    return est_grad

t1 = time.perf_counter()
for i in range(50):    
    
    # Initialize the cost
    init_cost = tf.dynamic_stitch(loss_gradient_demand(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, lambda_cost, lambda_flow_bal, lambda_positve).idx, access_cost)
    # Implement the BFGS optimization
    demand_opt = tfp.optimizer.bfgs_minimize(value_and_gradients_function=loss_gradient_demand(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, lambda_cost, lambda_flow_bal, lambda_positve), 
                                                initial_position=init_cost, tolerance=1e-08, max_iterations=500)
    # Assign the optimal solution
    access_cost = demand_opt.position

    # initialize the path flows
    init_path = tf.dynamic_stitch(loss_gradient_supply(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, lambda_cost, lambda_flow_bal, lambda_positve).idx, path_flow)
    # Implement the BFGS optimization
    supply_opt = tfp.optimizer.bfgs_minimize(value_and_gradients_function=loss_gradient_supply(access_cost, path_flow, path_link_inc, path_link_inc_n, od_path_inc, lambda_cost, lambda_flow_bal, lambda_positve), 
                                              initial_position=init_path, tolerance=1e-08, max_iterations=500)
    # Assign the optimal solution
    path_flow = supply_opt.position
    
    # Update the Lagrangian multipliers
    updated_lambdas = lambda_condi(access_cost, path_flow, 
                                  lambda_cost, 
                                  lambda_flow_bal, 
                                  lambda_positve, 
                                  path_link_inc, path_link_inc_n, od_path_inc)
    
    # Update the states
    lambda_cost = updated_lambdas[0]
    lambda_flow_bal = updated_lambdas[1]
    lambda_positve = updated_lambdas[2]

    print('iteration_step:', i)
    print('1. Updating the cost consistency:', access_cost)
    print('2. Updating the optimal path flows:', path_flow)
    print()

    updated_cost = []
    for i in range(access_cost.shape[0]):
      gen_val = tf.Variable(access_cost[i], dtype=tf.float32)
      updated_cost.append(gen_val)
      

    updated_path = []
    for i in range(path_flow.shape[0]):
      gen_val = tf.Variable(path_flow[i], dtype=tf.float32)
      updated_path.append(gen_val)
    
    access_cost = updated_cost
    path_flow   = updated_path

t2 = time.perf_counter()
print('Running Time:', t2-t1)