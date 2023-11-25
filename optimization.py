import tensorflow as tf
from data import data_generation
from objective_function import calculateCoreVars, optimizeSupply
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def admm_optimization(od_volume,
                      spare_od_path_inc,
                      path_link_inc,
                      path_link_inc_n,
                      path_flow,
                      bpr_params,
                      lagrangian_params,
                      lambda_positive,
                      training_steps):

    relative_gap = []
    for i in range(training_steps):
        logging.info("find optimal path flows")
        path_cost_, path_flow_n_, path_cost_n_, path_flow_all_, link_flow_, link_cost_ = \
            calculateCoreVars(path_flow,
                              od_volume,
                              spare_od_path_inc,
                              path_link_inc,
                              path_link_inc_n,
                              bpr_params,
                              )

        logging.info("compute minimum costs")
        ue_cost0 = []

        for i in range(len(path_cost_n_)):
            od_pair = tf.sparse.to_dense(tf.sparse.slice(spare_od_path_inc, [i, 0], [1, len(path_cost_)]))
            min_cost = tf.reduce_min(
                tf.multiply(tf.where(od_pair == 0, 1.0, 0) * 1e5 + od_pair, tf.reshape(path_cost_, (-1))))
            ue_cost0.append(min_cost)
        ue_cost0 = tf.reshape(tf.stack(ue_cost0), (tf.stack(ue_cost0).shape[0], 1))
        ue_cost = tf.reshape(tf.reduce_min(tf.concat([ue_cost0, path_cost_n_], axis=1), axis=1), (-1, 1))

        # calculate the relative gap
        comp_gap = tf.reduce_sum(tf.reshape(path_flow, (path_flow.shape[0], 1)) *
                                 (path_cost_ - tf.transpose(tf.sparse.sparse_dense_matmul(tf.transpose(ue_cost),
                                                                                          spare_od_path_inc)))) / \
                   tf.reduce_sum(tf.reshape(path_flow, (path_flow.shape[0], 1)) * path_cost_) \
                   + tf.reduce_sum(path_flow_n_ * (path_cost_n_ - ue_cost)) / tf.reduce_sum(path_flow_n_ * path_cost_n_)

        relative_gap.append(comp_gap.numpy())
        logging.info("relative gap: %s", comp_gap.numpy())

        path_flow = optimizeSupply(path_flow,
                                   lambda_positive,
                                   bpr_params,
                                   lagrangian_params,
                                   od_volume,
                                   spare_od_path_inc,
                                   path_link_inc,
                                   path_link_inc_n,)

        logging.info("optimal path flows: %s", path_flow.numpy())
        logging.info("update the Lagrangian multipliers")
        lambda_positive += lagrangian_params["rho_factor"] * lagrangian_params["rho_factor"] * (tf.nn.relu(-path_flow_all_))


if __name__ == "__main__":
    dir_data = './data/Sioux_Falls/'
    load_data = data_generation(dir_data)
    od_volume, spare_od_path_inc, path_link_inc, path_link_inc_n, _, = load_data.reformed_incidence_mat()
    path_flow = load_data.get_init_path_values(init_given=True)
    bpr_params = load_data.get_bpr_params()
    lagrangian_params, lambda_positive = load_data.get_lagrangian_params(path_link_inc_n, path_link_inc)
    training_steps = 10
    admm_optimization(od_volume,
                      spare_od_path_inc,
                      path_link_inc,
                      path_link_inc_n,
                      path_flow,
                      bpr_params,
                      lagrangian_params,
                      lambda_positive,
                      training_steps)