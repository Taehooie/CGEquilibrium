import pandas as pd
from functools import partial
import time
import tensorflow as tf
import tensorflow_probability as tfp
from data import data_generation
import pdb

# load dataset stored in the given folder
dir_data = './data/'
load_data = data_generation(dir_data)
od_path1_idx_list, path1_link_idx_list, path2_link_idx_list = load_data.incidence_mat()

num_link = load_data.link_df.shape[0]
od_volume = tf.reshape(tf.constant(load_data.od_df['volume'], dtype=tf.float32), (-1, 1))
od_path_inc = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(od_path1_idx_list, [1.0]*len(od_path1_idx_list), (len(od_volume),od_path1_idx_list[-1][1]+1))))
path_link_inc = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(path1_link_idx_list, [1.0]*len(path1_link_idx_list), (path1_link_idx_list[-1][0]+1, num_link))))
path_link_inc_n = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(path2_link_idx_list, [1.0]*len(path2_link_idx_list), (path2_link_idx_list[-1][0]+1, num_link))))

# Transit Costs
transit_cost = tf.reshape(tf.constant(load_data.od_df['travel_time'] * 1.2, dtype=tf.float32), (-1, 1))
fftt = tf.reshape(tf.constant(load_data.link_df['fftt'], dtype=tf.float32), (-1, 1))
cap = tf.reshape(tf.constant(load_data.link_df['capacity'], dtype=tf.float32), (-1, 1))

# Initialize the variables
theta = 0.1
bpr_alpha = 0.15
bpr_beta = 4
rho_ = 1.0

lambda_cost = tf.zeros((len(od_path_inc), 1), tf.float32)
lambda_flow_bal = tf.zeros((len(od_path_inc), 1), tf.float32)
lambda_positve = tf.zeros((len(path_link_inc_n) + len(path_link_inc), 1), tf.float32)
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


# def calculateLoss1(access_cost_, path_flow_, lambda_cost_, lambda_flow_bal_, lambda_positve_):
#
#     path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost_, path_flow_)
#
#     loss = tf.reduce_sum(path_flow_ * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) + tf.reduce_sum(tf.transpose(path_flow_n) * (path_cost_n - ue_cost))\
#            + tf.reduce_sum(tf.multiply(lambda_cost_, cost_condi)) + tf.reduce_sum((rho_/2)*cost_condi**2)\
#            + tf.reduce_sum(tf.multiply(lambda_flow_bal_, flow_condi)) + tf.reduce_sum((rho_/2)*flow_condi**2)\
#            + tf.reduce_sum(tf.multiply(lambda_positve_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum((rho_/2)*tf.nn.relu(-path_flow_all)**2)
#     return loss
#
# def calculateLoss2(path_flow_, access_cost_, lambda_cost_, lambda_flow_bal_, lambda_positve_):
#
#     path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost_, path_flow_)
#
#     loss = tf.reduce_sum(path_flow_ * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) + \
#            tf.reduce_sum(tf.transpose(path_flow_n) * (path_cost_n - ue_cost))\
#            + tf.reduce_sum(tf.multiply(lambda_cost_, cost_condi)) + tf.reduce_sum((rho_/2)*cost_condi**2)\
#            + tf.reduce_sum(tf.multiply(lambda_flow_bal_, flow_condi)) + tf.reduce_sum((rho_/2)*flow_condi**2)\
#            + tf.reduce_sum(tf.multiply(lambda_positve_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum((rho_/2)*tf.nn.relu(-path_flow_all)**2)
#     return loss

def calculateLoss1(access_cost_, path_flow_, lambda_cost_, lambda_flow_bal_, lambda_positve_):

    path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost_, path_flow_)

    loss = tf.reduce_sum(tf.reshape(path_flow_, (1043, 1)) * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) \
           + tf.reduce_sum(path_flow_n * (path_cost_n - ue_cost))\
           + tf.reduce_sum(tf.multiply(lambda_cost_, cost_condi)) + tf.reduce_sum((rho_/2)*cost_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_flow_bal_, flow_condi)) + tf.reduce_sum((rho_/2)*flow_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_positve_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum((rho_/2)*tf.nn.relu(-path_flow_all)**2)
    return loss

def calculateLoss2(path_flow_, access_cost_, lambda_cost_, lambda_flow_bal_, lambda_positve_):

    path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost_, path_flow_)

    loss = tf.reduce_sum(tf.reshape(path_flow_, (1043, 1)) * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) + \
           tf.reduce_sum(path_flow_n * (path_cost_n - ue_cost))\
           + tf.reduce_sum(tf.multiply(lambda_cost_, cost_condi)) + tf.reduce_sum((rho_/2)*cost_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_flow_bal_, flow_condi)) + tf.reduce_sum((rho_/2)*flow_condi**2)\
           + tf.reduce_sum(tf.multiply(lambda_positve_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum((rho_/2)*tf.nn.relu(-path_flow_all)**2)
    return loss

def relative_gap(path_flow_, access_cost_):

    path_cost, path_flow_n, path_cost_n, ue_cost, cost_condi, flow_condi, path_flow_all = calculateCoreVars(access_cost_, path_flow_)
    rel_gap = tf.reduce_sum(tf.reshape(path_flow_, (1043, 1)) * (path_cost - tf.matmul(tf.transpose(od_path_inc), ue_cost))) \
              / tf.reduce_sum(tf.reshape(path_flow_, (1043, 1)) * path_cost) + \
              tf.reduce_sum(path_flow_n * (path_cost_n - ue_cost)) / tf.reduce_sum(path_flow_n * path_cost_n)

    return rel_gap

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

relative_gap_store = []
for i in range(10):
    access_cost = optimizeDemand(access_cost, path_flow, lambda_cost, lambda_flow_bal, lambda_positve)
    path_flow = optimizeSupply(path_flow, access_cost, lambda_cost, lambda_flow_bal, lambda_positve)

    # Update the Lagrangian multipliers
    _, _, _, _, cost_condi_, flow_condi_, path_flow_all_ = calculateCoreVars(access_cost, path_flow)
    lambda_cost += rho_ * cost_condi_
    lambda_flow_bal += rho_ * flow_condi_
    lambda_positve += rho_ * rho_ * (tf.nn.relu(-path_flow_all_))

    rel_gap = relative_gap(path_flow, access_cost)
    relative_gap_store.append(rel_gap.numpy())
    print('iteration_step:', i)
    # print('1. Updating the cost consistency:', access_cost)
    # print('2. Updating the optimal path flows:', path_flow)
    # TODO: path_volume*(c-pie) / (path_volume*c)
    print('Relative Gap:', rel_gap)

pdb.set_trace()
t2 = time.perf_counter()
print('Running Time:', t2-t1)