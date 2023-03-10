import pandas as pd
from functools import partial
import time
import tensorflow as tf
import tensorflow_probability as tfp
import os

# Load the datasets
data_folder = 'data'
agent1 = pd.read_csv(os.path.join(data_folder, 'agent_type_1_v2.csv'))
agent2 = pd.read_csv(os.path.join(data_folder, 'agent_type_2_v2.csv'))
agent3 = pd.read_csv(os.path.join(data_folder, 'agent_type_3_v2.csv'))
agent4 = pd.read_csv(os.path.join(data_folder, 'agent_type_4_v2.csv'))

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
link_df['link_no'] = list(range(len(link_df)))
link_no_pair_dict = link_df[['link_no', 'link_pair']].set_index('link_pair').to_dict()['link_no']

# Origin-Destination to Origin
od_ozone_dict = od_df[['ozone_id', 'od_id']].set_index('od_id').to_dict()['ozone_id']

# 1. Generate the orgin-destination volume
od_volume = tf.reshape(tf.constant(od_df['od_demand'], dtype=tf.float32), (-1,1))

# 2. Generate the od-path incidence matrix
# To count the number of x_f flow variables

path_no1 = 0
od_path1_idx_list = []
path1_link_idx_list = []
path2_link_idx_list = []
for i in range(len(od_df)):
    od_id = od_df.loc[i, 'od_id']
    path_df_od = path_df[path_df['od_id'] == od_id].reset_index(drop=True)
    for j in range(len(path_df_od)):
        node_sequence = list(map(int, path_df_od.loc[j,'node_sequence'].split(';')))
        link_sequence = [link_no_pair_dict[(node_sequence[k], node_sequence[k+1])] for k in range(len(node_sequence)-1)]

        if j < len(path_df_od) - 1:
            od_path1_idx_list.append((i,path_no1))
            for link_id in link_sequence:
                path1_link_idx_list.append((path_no1, link_id))
            path_no1 += 1
        else:
            for link_id in link_sequence:
                path2_link_idx_list.append((i, link_id))

od_path_inc = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(od_path1_idx_list, [1.0]*len(od_path1_idx_list), (len(od_volume),od_path1_idx_list[-1][1]+1))))
path_link_inc = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(path1_link_idx_list, [1.0]*len(path1_link_idx_list), (path1_link_idx_list[-1][0]+1, num_link))))
path_link_inc_n = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(path2_link_idx_list, [1.0]*len(path2_link_idx_list), (path2_link_idx_list[-1][0]+1, num_link))))


# Transit Costs
transit_cost = tf.reshape(tf.constant(od_df['od_time'] * 1.2, dtype=tf.float32), (-1, 1))
fftt = tf.reshape(tf.constant(link_df['VDF_fftt1'], dtype=tf.float32), (-1, 1))
cap = tf.reshape(tf.constant(link_df['VDF_cap1'], dtype=tf.float32), (-1, 1))

# Initialize the variables
theta = 0.1
bpr_alpha = 0.15
bpr_beta = 4

lambda_cost = tf.zeros((len(od_path_inc), 1), tf.float32)
lambda_flow_bal = tf.zeros((len(od_path_inc), 1), tf.float32)
lambda_positve = tf.zeros((len(path_link_inc_n) + len(path_link_inc), 1), tf.float32)
rho_ = 1.0

od_path_inc_for_min = (tf.ones(tf.shape(od_path_inc)) - od_path_inc) * 1e5 + od_path_inc

@tf.function
def calculateCoreVars(access_cost_, path_flow_):
    access_cost_ = tf.reshape(access_cost_, (-1,1))
    path_flow_ = tf.reshape(path_flow_, (-1, 1))

    auto_volume = od_volume * tf.exp(-theta*access_cost_) / (tf.exp(-theta*access_cost_) + tf.exp(-theta*transit_cost))
    path_flow_n = auto_volume - tf.matmul(od_path_inc, path_flow_)

    link_flow = tf.matmul(tf.transpose(path_link_inc), path_flow_) + tf.matmul(tf.transpose(path_link_inc_n), path_flow_n)
    link_cost = fftt*(1 + bpr_alpha*(link_flow/cap)**bpr_beta)

    path_cost = tf.matmul(path_link_inc, link_cost)
    path_cost_n = tf.matmul(path_link_inc_n, link_cost)

    ue_cost0 = tf.reshape(tf.reduce_min(tf.multiply(od_path_inc_for_min, tf.reshape(path_cost, (-1,))), axis=1), (-1,1))
    ue_cost = tf.reshape(tf.reduce_min(tf.concat([ue_cost0, path_cost_n], axis=1), axis=1), (-1,1))

    path_flow_all = tf.concat([path_flow_, path_flow_n], axis=0)

    flow_condi = auto_volume - path_flow_n
    cost_condi = access_cost_ - ue_cost

    return path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all


def calculateLoss1(access_cost_, path_flow_, lambda_cost_, lambda_flow_bal_, lambda_positve_):

    path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost_, path_flow_)

    loss = tf.reduce_sum(path_flow_ * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) + tf.reduce_sum(tf.transpose(path_flow_n) * (path_cost_n - ue_cost))\
           + tf.reduce_sum(tf.multiply(lambda_cost_, cost_condi)) + tf.reduce_sum((rho_/2)*cost_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_flow_bal_, flow_condi)) + tf.reduce_sum((rho_/2)*flow_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_positve_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum((rho_/2)*tf.nn.relu(-path_flow_all)**2)
    return loss

def calculateLoss2(path_flow_, access_cost_, lambda_cost_, lambda_flow_bal_, lambda_positve_):

    path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost_, path_flow_)

    loss = tf.reduce_sum(path_flow_ * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) + tf.reduce_sum(tf.transpose(path_flow_n) * (path_cost_n - ue_cost))\
           + tf.reduce_sum(tf.multiply(lambda_cost_, cost_condi)) + tf.reduce_sum((rho_/2)*cost_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_flow_bal_, flow_condi)) + tf.reduce_sum((rho_/2)*flow_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_positve_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum((rho_/2)*tf.nn.relu(-path_flow_all)**2)
    return loss

@tf.function
def optimizeDemand(initial_cost, path_flow_, lambda_cost_, lambda_flow_bal_, lambda_positve_):

    calculateLossDemand = partial(calculateLoss1, path_flow_=path_flow_, lambda_cost_=lambda_cost_, lambda_flow_bal_=lambda_flow_bal_, lambda_positve_=lambda_positve_)

    def loss_gradient_demand(access_cost_):  # access_cost_ as variables
        return tfp.math.value_and_gradient(calculateLossDemand, access_cost_)

    demand_opt = tfp.optimizer.bfgs_minimize(value_and_gradients_function=loss_gradient_demand, initial_position=initial_cost, tolerance=1e-08, max_iterations=500)
    return demand_opt.position


@tf.function
def optimizeSupply(initial_path, access_cost_, lambda_cost_, lambda_flow_bal_, lambda_positve_):

    calculateLossSupply = partial(calculateLoss2, access_cost_=access_cost_, lambda_cost_=lambda_cost_, lambda_flow_bal_=lambda_flow_bal_, lambda_positve_=lambda_positve_)

    def loss_gradient_supply(path_flow_):  # path_flow_ as variables
        return tfp.math.value_and_gradient(calculateLossSupply, path_flow_)

    supply_opt = tfp.optimizer.bfgs_minimize(value_and_gradients_function=loss_gradient_supply, initial_position=initial_path, tolerance=1e-08, max_iterations=500)
    return supply_opt.position


t1 = time.perf_counter()

access_cost = tf.Variable(tf.random.uniform([od_path_inc.shape[0]], minval=0, maxval=10))
path_flow = tf.Variable(tf.random.uniform([path_link_inc.shape[0]], minval=0, maxval=10))

for i in range(50):
    access_cost = optimizeDemand(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve)
    path_flow = optimizeSupply(path_flow, access_cost, lambda_cost, lambda_flow_bal, lambda_positve)

    # Update the Lagrangian multipliers
    _, _, _, _, cost_condi_, flow_condi_, path_flow_all_ = calculateCoreVars(access_cost, path_flow)
    lambda_cost += rho_ * cost_condi_
    lambda_flow_bal += rho_ * flow_condi_
    lambda_positve += rho_ * rho_ * (tf.nn.relu(-path_flow_all_))

    print('iteration_step:', i)
    print('1. Updating the cost consistency:', access_cost)
    print('2. Updating the optimal path flows:', path_flow)
    print()

t2 = time.perf_counter()
print('Running Time:', t2-t1)