import jax.numpy as jnp
from scipy import optimize
from jax import jit, grad


theta = 0.1
bpr_alpha = 0.15
bpr_beta = 4


od_volume = jnp.array([[2000], [3000]])

od_path_inc = jnp.array([[1, 1, 0],
                         [0, 0, 1]])
path_link_inc = jnp.array([[1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 1, 1]])
path_link_inc_n = jnp.array([[1, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1]])

transit_cost = jnp.array([[20], [25]])
fftt = jnp.array([[5], [6], [5], [4], [6], [3]])
cap = jnp.array([[5000], [6000], [5000], [4000], [6000], [3000]])


od_path_inc_for_min = (jnp.ones(jnp.shape(od_path_inc)) - od_path_inc) * 1e5 + od_path_inc



# @jit
def calculateLoss(access_cost, path_flow):
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

    loss = jnp.sum(path_flow * (path_cost - jnp.matmul(od_path_inc.T, ue_cost))) + jnp.sum(path_flow_n.T * (path_cost_n - ue_cost))

    return loss * 1e-3




access_cost_ = jnp.array([10,20])
path_flow_ = jnp.array([100,200,150])

d = calculateLoss(access_cost_, path_flow_)

res = optimize.minimize(calculateLoss,
                        access_cost_,
                        jac=jit(grad(calculateLoss)),
                        args=(path_flow_,),
                        method='BFGS')


print('done')