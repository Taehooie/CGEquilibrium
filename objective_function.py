from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def calculateCoreVars(path_flow_,
                      od_volume,
                      spare_od_path_inc,
                      path_link_inc,
                      path_link_inc_n,
                      bpr_params,):

    path_flow_ = tf.reshape(path_flow_, (-1, 1))
    path_flow_n = od_volume - tf.sparse.sparse_dense_matmul(spare_od_path_inc, path_flow_)

    link_flow = tf.matmul(tf.transpose(path_link_inc), path_flow_) + tf.matmul(tf.transpose(path_link_inc_n),
                                                                               path_flow_n)
    link_cost = bpr_params["fftt"] * (1 + bpr_params["alpha"] * (link_flow / bpr_params["cap"]) ** bpr_params["beta"])

    path_cost = tf.matmul(path_link_inc, link_cost)
    path_cost_n = tf.matmul(path_link_inc_n, link_cost)

    path_flow_all = tf.concat([path_flow_, path_flow_n], axis=0)

    return path_cost, path_flow_n, path_cost_n, path_flow_all, link_flow, link_cost


def calculateLoss(path_flow_,
                  od_volume,
                  spare_od_path_inc,
                  path_link_inc,
                  path_link_inc_n,
                  lambda_positive_,
                  bpr_params,
                  lagrangian_params):

    path_cost, path_flow_n, path_cost_n, path_flow_all, link_flow, link_cost = \
        calculateCoreVars(path_flow_,
                          od_volume,
                          spare_od_path_inc,
                          path_link_inc,
                          path_link_inc_n,
                          bpr_params)

    loss = tf.reduce_sum(
        bpr_params["fftt"] * link_flow + (bpr_params["alpha"] * bpr_params["fftt"]) /
        (bpr_params["beta"] + 1) * (link_flow / bpr_params["cap"]) ** (bpr_params["beta"] + 1) * bpr_params["cap"]) \
           + tf.reduce_sum(tf.multiply(lambda_positive_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum(
        (lagrangian_params["rho_factor"] / 2) * tf.nn.relu(-path_flow_all) ** 2)
    return loss


@tf.function
def optimizeSupply(initial_path,
                   lambda_positive_,
                   bpr_params,
                   lagrangian_params,
                   od_volume,
                   spare_od_path_inc,
                   path_link_inc,
                   path_link_inc_n,):

    calculateLossSupply = partial(calculateLoss,
                                  od_volume=od_volume,
                                  spare_od_path_inc=spare_od_path_inc,
                                  path_link_inc=path_link_inc,
                                  path_link_inc_n=path_link_inc_n,
                                  lambda_positive_=lambda_positive_,
                                  bpr_params=bpr_params,
                                  lagrangian_params=lagrangian_params)
    # commenting argument is a key to check which arguments will be partial

    def loss_gradient_supply(path_flow_):
        return tfp.math.value_and_gradient(calculateLossSupply, path_flow_)

    # TODO: check the convergence
    supply_opt = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=loss_gradient_supply,
                                              initial_position=initial_path,
                                              tolerance=1e-08,
                                              max_iterations=500)
    return supply_opt.position
