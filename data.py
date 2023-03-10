import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import time
from functools import partial

import pdb

class data_generation():
    def __init__(self, dir_path):

        self.route_assignment_data = pd.read_csv(str(dir_path) + "route_assignment.csv")
        self.link_performance_data = pd.read_csv(str(dir_path) + "link_performance.csv")

        # origin data
        self.ozone_df = self.route_assignment_data.groupby('o_zone_id')['volume'].sum()
        self.ozone_df = self.ozone_df.reset_index()
        self.ozone_df['o_node_id'] = self.ozone_df['o_zone_id']

        # origin-destination data
        self.od_df = self.route_assignment_data.groupby(['o_zone_id', 'd_zone_id'])['volume', 'travel_time'].sum()
        self.od_df = self.od_df.reset_index()
        self.od_df['od_id'] = self.od_df.index + 1

        # path data
        self.path_df = self.route_assignment_data[['o_zone_id', 'd_zone_id', 'node_sequence']]
        self.path_df['path_id'] = self.path_df.index + 1

        # link data
        self.link_df = self.link_performance_data[['link_id', 'from_node_id', 'to_node_id', 'fftt', 'capacity']]
        self.link_df['link_no'] = self.link_df.index

    def origin_layer(self):
        # origin layer
        node_zone_dict = self.ozone_df[['o_node_id', 'o_zone_id']].set_index('o_node_id').to_dict()['o_zone_id']
        return node_zone_dict

    def origin_dest_layer(self):
        # origin-destination layer
        node_zone_dict = self.origin_layer()
        self.od_df['od_pair'] = self.od_df.apply(lambda x: (int(x.o_zone_id), int(x.d_zone_id)), axis=1)
        self.od_df['o_zone_id'] = self.od_df.apply(lambda x: node_zone_dict[int(x.o_zone_id)], axis=1)
        od_pair_dict = self.od_df[['od_pair', 'od_id']].set_index('od_pair').to_dict()['od_id']
        return od_pair_dict

    def od_to_path_layer(self):
        od_pair_dict = self.origin_dest_layer()
        self.path_df['od_id'] = self.path_df.apply(lambda x: od_pair_dict[int(x.o_zone_id), int(x.d_zone_id)], axis=1)
        # path_od_dict = self.path_df[['path_id', 'od_id']].set_index('path_id').to_dict()['od_id']
        return self.path_df

    def path_link_layer(self):
        self.link_df['link_pair'] = self.link_df.apply(lambda x: (int(x.from_node_id), int(x.to_node_id)), axis=1)
        link_no_pair_dict = self.link_df[['link_no', 'link_pair']].set_index('link_pair').to_dict()['link_no']
        return link_no_pair_dict

    def incidence_mat(self):
        # To count the number of x_f flow variables
        link_no_pair_dict = self.path_link_layer()
        path_df = self.od_to_path_layer()
        path_no1 = 0
        od_path1_idx_list = []
        path1_link_idx_list = []
        path2_link_idx_list = []
        for i in range(len(self.od_df)):
            # pdb.set_trace()
            od_id = self.od_df.loc[i, 'od_id']
            path_df_od = path_df[path_df['od_id'] == od_id].reset_index(drop=True)

            for j in range(len(path_df_od)):
                node_sequence = list(map(int, path_df_od.loc[j, 'node_sequence'].split(';')[0: -1]))
                link_sequence = [link_no_pair_dict[(node_sequence[k], node_sequence[k + 1])] for k in range(len(node_sequence) - 1)]

                if j < len(path_df_od) - 1:
                    od_path1_idx_list.append((i, path_no1))
                    for link_id in link_sequence:
                        path1_link_idx_list.append((path_no1, link_id))
                    path_no1 += 1
                else:
                    for link_id in link_sequence:
                        path2_link_idx_list.append((i, link_id))

        return od_path1_idx_list, path1_link_idx_list, path2_link_idx_list


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